import torch
from sklearn.model_selection import train_test_split

import sys
import os



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

def split_data(data, ratio = []):
    node_num = data.x.shape[0]
    label = data.y.numpy()
    test_size = ratio[1]/sum(ratio)
    val_size = ratio[2]/(ratio[0]+ratio[2])


    train_mask, test_mask, train_label, test_label = train_test_split([i for i in range(node_num)], label,
                                                                      test_size=test_size, stratify=label, random_state=200)
    train_mask, validation_mask, _, _ = train_test_split(train_mask, train_label, test_size=val_size, stratify=train_label,
                                                         random_state=100)
    return train_mask, validation_mask, test_mask

def train(*args):
    model, optimizer, data, device, criterion, mask = args
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(out[mask], data.y[mask].to(device))
    loss.backward()
    optimizer.step()
    model.eval()
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask].to(device)).sum()
    acc = int(correct) / len(mask)
    return loss.item(), acc

# 검증 함수 정의
def validate(*args):
    model, optimizer, data, device, criterion, mask = args
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    # print(model.get_emb(data.x.to(device), data.edge_index.to(device)))
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask].to(device)).sum()
    acc = int(correct) / len(mask)
    return acc

# 테스트 함수 정의
def test(*args):
    model, optimizer, data, device, criterion, mask = args
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask].to(device)).sum()
    acc = int(correct) / len(mask)
    return acc

def total(*args):
    model, optimizer, data, device, criterion = args
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    correct = (pred == data.y.to(device)).sum()
    acc = int(correct) / data.x.shape[0]
    return acc