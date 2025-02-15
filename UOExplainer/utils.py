import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from arguments import args
from torch.utils.data import Dataset

sys.path.append('../training_gnn/')

def load_data(data):
    data_path = f'../dataset/{data}/'
    edge_list = open(f'{data_path}edge_list.txt', 'r')
    edge = [[], []]
    for line in edge_list:
        node1, node2 = map(int, line.strip().split(' '))
        edge[0].append(node1)
        edge[1].append(node2)
    edge = torch.LongTensor(edge)

    feature_list = open(f'{data_path}features.txt', 'r')
    feature_list = feature_list.readlines()
    feature_dim = len(feature_list[0].strip().split(' ')) -2
    n_node = len(feature_list)
    label = torch.zeros((n_node))
    feature = torch.zeros((n_node,feature_dim ))

    for features in feature_list:
        features = list(map(float, features.strip().split()))
        node, features_, label_ = features[0], features[1:-1], features[-1]

        label[int(node)] = label_
        feature[int(node), :] = torch.FloatTensor(features_)
    return feature, label.long(), edge.long()

def load_model(data):
    data_path = f'../dataset/{data}/'
    model_path = [i for i in os.listdir(data_path) if i.endswith('00')]
    if len(model_path) == 1:
        model_path = f'{data_path}{model_path[0]}/'
    model = torch.load(f'{model_path}model.pth')
    return model, model_path



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the probability of the target class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def make_graph(data):
    graph = nx.Graph()
    edge_list = open(f'../dataset/{data}/edge_list.txt', 'r')
    for i in edge_list:
        node1, node2 = map(int, i.strip().split(' '))
        graph.add_edge(node1, node2)
    return graph

def undi_di(data):
    data_path = f'../dataset/{data}/'
    edge_list = open(f'{data_path}edge_list.txt', 'r')
    edge_txt = open(f'{data_path}edge_list_for_counting.txt', 'w')
    for line in edge_list:
        node1, node2 = map(int, line.strip().split(' '))
        if node1<node2:
            edge_txt.write(f'{node1}\t{node2}\n')








