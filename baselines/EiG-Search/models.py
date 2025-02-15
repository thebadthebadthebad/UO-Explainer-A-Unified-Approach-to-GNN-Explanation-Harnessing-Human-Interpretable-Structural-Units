from torch_geometric.nn import GCNConv, BatchNorm
import torch
import torch.nn as nn
import torch.nn.init as init

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()


class IdenticalPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return x

class GCN_3l_norm(torch.nn.Module):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes, num_layer=3):
        super().__init__()
        num_layer = 3
        self.embedding_size = dim_hidden
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.ba1 = BatchNorm(dim_hidden, momentum=0.99)
        self.relu1 = torch.nn.ReLU()
        self.relus = torch.nn.ModuleList(
            [
                torch.nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.bas = torch.nn.ModuleList(
            [
                BatchNorm(dim_hidden, momentum=0.99)
                for _ in range(num_layer - 1)
            ]
        )

        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            print('We are not provide the graph classification version')

        self.ffn = torch.nn.Sequential(*(
                [torch.nn.Dropout(p=0.4), torch.nn.Linear(dim_hidden, num_classes)]
        ))
        self.reset_parameters()
        #self.dropout = torchnn.Dropout()

    def forward(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weights)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weights)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return out, out_readout

    def embedding(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weights)))
        for conv, ba, relu in zip(self.convs,self.bas, self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weights)))
        return post_conv

    def reset_parameters(self):

        init.xavier_uniform_(self.conv1.lin.weight)
        for conv in self.convs:
            init.xavier_uniform_(conv.lin.weight)
        init.xavier_uniform_(self.ffn[-1].weight)
        #init.xavier_uniform_(self.conv2.weight)

class GCN_3l_Gelu(torch.nn.Module):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes, num_layer=3):
        super().__init__()
        num_layer = 3
        self.embedding_size = dim_hidden
        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.ba1 = BatchNorm(dim_hidden, momentum=0.99)
        self.relu1 = torch.nn.GELU()
        self.relus = torch.nn.ModuleList(
            [
                torch.nn.GELU()
                for _ in range(num_layer - 1)
            ]
        )
        self.bas = torch.nn.ModuleList(
            [
                BatchNorm(dim_hidden, momentum=0.99)
                for _ in range(num_layer - 1)
            ]
        )

        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            print('We are not provide the graph classification version')

        self.ffn = torch.nn.Sequential(*(
                [torch.nn.Dropout(p=0.3), torch.nn.Linear(dim_hidden, num_classes)]
        ))
        self.reset_parameters()
        #self.dropout = torchnn.Dropout()

    def forward(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weights)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weights)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return out, out_readout

    def embedding(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weights)))
        for conv, ba, relu in zip(self.convs,self.bas, self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weights)))
        return post_conv

    def reset_parameters(self):

        init.xavier_uniform_(self.conv1.lin.weight)
        for conv in self.convs:
            init.xavier_uniform_(conv.lin.weight)
        init.xavier_uniform_(self.ffn[-1].weight)
        #init.xavier_uniform_(self.conv2.weight)

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class IdenticalPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class GCN_3l_agg(torch.nn.Module):

    def __init__(self, model_level, dim_node, dim_hidden, num_classes,num_layer=3):
        super().__init__()
        num_layer = 3

        self.conv1 = GCNConv(dim_node, dim_hidden)
        self.convs = torch.nn.ModuleList(
            [
                GCNConv(dim_hidden, dim_hidden)
                for _ in range(num_layer - 1)
             ]
        )
        self.ba1 = BatchNorm(dim_hidden, momentum=0.99)
        self.relu1 = torch.nn.ReLU()
        self.relus = torch.nn.ModuleList(
            [
                torch.nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        self.bas = torch.nn.ModuleList(
            [
                BatchNorm(dim_hidden, momentum=0.99)
                for _ in range(num_layer - 1)
            ]
        )

        if model_level == 'node':
            self.readout = IdenticalPool()
        else:
            print('We are not provide the graph classification version')

        self.ffn = torch.nn.Sequential(*(
                [torch.nn.Dropout(p=0.2), torch.nn.Linear(dim_hidden*3, num_classes)]
        ))
        self.reset_parameters()
        #self.dropout = torchnn.Dropout()

    def forward(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        stack = []
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index)))
        stack.append(post_conv)
        for conv, ba, relu in zip(self.convs,self.bas, self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index)))
            stack.append(post_conv)
        #print(stack)
        post_conv = torch.cat(stack, dim=1)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return out

    def get_emb(self, x, edge_index, edge_weights=None) -> torch.Tensor:
        stack = []
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index)))
        stack.append(post_conv)
        for conv, ba, relu in zip(self.convs,self.bas, self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index)))
            stack.append(post_conv)
        post_conv = torch.cat(stack, dim=1)
        return post_conv

    def reset_parameters(self):

        init.xavier_uniform_(self.conv1.lin.weight)
        for conv in self.convs:
            init.xavier_uniform_(conv.lin.weight)
        init.xavier_uniform_(self.ffn[-1].weight)
        #init.xavier_uniform_(self.conv2.weight)

