#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:53:34 2020

@author: sohailnizam
"""


import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import time
from datetime import datetime
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import convert
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 1

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
            #return pyg_nn.SAGEConv(input_dim, hidden_dim)
            #return pyg_nn.GraphConv(input_dim, hidden_dim)
            #return pyg_nn.ARMAConv(input_dim, hidden_dim)
            #return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
             #                     nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)
                
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x
            #x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)
        
        #x = self.post_mp(x)
        

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
    
    


def train(dataset, task, writer):
    test_acc_list = []
    if task == 'graph':
        data_size = len(dataset)
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build model
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0
        model.train()
        for batch in loader:
            # print(batch.train_mask, '----')
            opt.zero_grad()
            embedding, pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_acc_list.append(test_acc)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))
            writer.add_scalar("test accuracy", test_acc, epoch)

    return model, test_acc_list


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            emb, pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item()
    return correct / total



pubmed = Planetoid(root='/tmp/pubmed', name='pubmed')
writer = SummaryWriter("./log/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
test_loader = DataLoader(pubmed, batch_size=64, shuffle=True)

#model 0: only 1 conv layer, no relu or dropout after
#no post_mp
#with GCNConv
node_model0, acc_list0 = train(pubmed, 'node', writer)
test_acc0 = test(test_loader, node_model0) #.741

#model 1: no relu or dropput after conv
# no post_mp
#with GCNConv
node_model1, acc_list1 = train(pubmed, 'node', writer)
test_acc1 = test(test_loader, node_model1) #.722

#model 2: model 1 plus relu and dropout after conv (still no post_mp)
#with GCNConv
node_model2, acc_list2 = train(pubmed, 'node', writer)
test_acc2 = test(test_loader, node_model2) #.749

#model 3: model 2 plus relu and dropout after conv
#plus two fully connected layers (post_mp) (this is the model from main.py)
node_model3, acc_list3 = train(pubmed, 'node', writer)
test_acc3 = test(test_loader, node_model3) #.7500

#model 4: model 3 with extra relu layer added between fully connected
node_model4, acc_list4 = train(pubmed, 'node', writer)
test_acc4 = test(test_loader, node_model4) #.742

#model 5: model 3 with SAGEConv instead
node_model5, acc_list5 = train(pubmed, 'node', writer)
test_acc5 = test(test_loader, node_model5) #.739

#model 6: model 3 with GINConv instead
node_model6, acc_list6 = train(pubmed, 'node', writer)
test_acc6 = test(test_loader, node_model6) #.772

#model 6 is the winner

#create a pd dataset to hold the test acc lists
acc_data = {'model0' : acc_list0,
            'model1' : acc_list1,
            'model2' : acc_list2,
            'model3' : acc_list3,
            'model4' : acc_list4,
            'model5' : acc_list5,
            'model6' : acc_list6}

acc_data = pd.DataFrame(acc_data)
acc_data.to_csv('./acc_data.csv', index=False)



#visualization
color_list = ["red", "green", "blue"]

loader = DataLoader(pubmed, batch_size=64, shuffle=True)
embs = []
colors = []
for batch in loader:
    emb, pred = node_model2(batch)
    embs.append(emb)
    colors += [color_list[y] for y in batch.y]
embs = torch.cat(embs, dim=0)

xs, ys = zip(*TSNE().fit_transform(embs.detach().numpy()))
plt.scatter(xs, ys, color=colors)