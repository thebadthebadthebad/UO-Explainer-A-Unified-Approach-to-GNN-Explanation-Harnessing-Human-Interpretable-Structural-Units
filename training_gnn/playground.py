import torch
import sys
import os
from utils import *
import torch_geometric
from models import GCN_3l_norm, GCN_3l_agg, GCN_3l_Gelu

def main():
    #Argments
    device = 2
    data_name = 'tree_grid'
    data_path = f'../dataset/{data_name}/'

    ratio = [8, 1, 1]
    hidden_d = 20
    learning_rate = 0.005
    w = 1e-5
    epochs = 2000



    feature, label, edge = load_data(data_name)

    class_n = torch.unique(label).shape[0]
    node_n = feature.shape[1]

    data = torch_geometric.data.Data(x=feature, edge_index=edge, y=label)
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')

    train_mask, val_mask, test_mask = split_data(data, ratio)
    if data_name in ['bashapes', 'tree_cycle']:
        model = GCN_3l_norm(model_level='node', dim_node=node_n, dim_hidden=hidden_d, num_classes=class_n).to(device)
    elif data_name in ['bac', 'tree_grid']:
        #model = GCN_3l_agg(model_level='node', dim_node=node_n, dim_hidden=hidden_d, num_classes=class_n).to(device)
        model = GCN_3l_Gelu(model_level='node', dim_node=node_n, dim_hidden=hidden_d, num_classes=class_n, num_layer=3).to(device)
    optimizer = torch.optim.Adam(lr=learning_rate, weight_decay=w, params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    if not(os.path.exists(f'{data_path}{learning_rate}_{epochs}')):
        os.mkdir(f'{data_path}{learning_rate}_{epochs}')

    global_acc = 0
    for epoch in range(1, epochs+1):
        model.train()

        loss, acc = train(model, optimizer, data, device, criterion, train_mask)
        print(acc)
        #print(model.ba1.module.running_mean)

        model.eval()
        #print(model.ba1.module.running_mean)
        val_acc = validate(model, optimizer, data, device, criterion, val_mask)
        print(val_acc)
        total_acc = total(model, optimizer, data, device, criterion)
        if total_acc > global_acc:
            global_acc = total_acc
            torch.save(model.to('cpu'), f'{data_path}{learning_rate}_{epochs}/model.pth')
            model.to(device)
        print(test(model, optimizer, data, device, criterion, test_mask))
        print('----------------')

    model = torch.load(f'{data_path}{learning_rate}_{epochs}/model.pth').to(device)
    model.eval()
    #print(model.ba1.module.running_mean)
    print(test(model, optimizer, data, device, criterion, train_mask))
    print(test(model, optimizer, data, device, criterion, val_mask))
    print(test(model, optimizer, data, device, criterion, test_mask))
    print(total(model, optimizer, data, device, criterion))
    model.to('cpu')

if __name__ == "__main__":
    main()


