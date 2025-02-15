from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn


class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class IdenticalPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

class GCN_3l(torch.nn.Module):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.relu1 = torch.nn.ReLU()
        self.relus = torch.nn.ModuleList(
            [
                torch.nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            print('We are not provide the graph classification version')

        self.ffn = torch.nn.Sequential(*(
                [torch.nn.Linear(dim_hidden, dim_hidden)] +
                [torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.Linear(dim_hidden, num_classes)]
        ))

        #self.dropout = torchnn.Dropout()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        x, edge_index, batch = self.arguments_read(*args, **kwargs)

        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))

        out_readout = self.readout(post_conv, batch)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, *args, **kwargs) -> torch.Tensor:
        x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.relu1(self.conv1(x, edge_index))
        for conv, relu in zip(self.convs, self.relus):
            post_conv = relu(conv(post_conv, edge_index))
        return post_conv

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class IdenticalPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x


