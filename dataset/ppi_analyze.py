import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader

import os
import argparse



path = './data/PPI'  # 데이터를 저장할 경로 설정

# PPI 데이터셋 로드
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
for i in train_loader:
    print(i)
print('yes')
for i in val_loader:
    print(i)
print('yes')
for i in test_loader:
    print(i)
print('yes')


feature = train_dataset[5].x
y = train_dataset[5].y
for i in range(6):
    c = open(f'ppi{i}/features.txt', 'w')
    for j in range(feature.shape[0]):
        j_feature = feature[j, :].tolist()
        #print(j_feature)
        j_y = y[j, i].long().item()
        line = f'{j} {" ".join(list(map(str, j_feature)))} {j_y}\n'
        c.write(line)
        print(line)

# print(feature)
# print(y)