from torch_geometric.nn import GCNConv, BatchNorm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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
                [torch.nn.Dropout(p=0.3, training=self.training), torch.nn.Linear(dim_hidden, num_classes)]
        ))
        #self.dropout = torchnn.Dropout()

    def forward(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return F.softmax(out, dim=-1)

    def forward2(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return out

    def fwd_eval(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return F.softmax(out, dim=-1)



    def fwd_cam(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))

        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return F.softmax(out, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

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
                [torch.nn.Dropout(p=0.3, training=self.training), torch.nn.Linear(dim_hidden, num_classes)]
        ))

        #self.dropout = torchnn.Dropout()

    def forward(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return  F.softmax(out, dim=-1)

    def forward(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return  out

    def fwd_cam(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return  F.softmax(out, dim=-1)

    def fwd_eval(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #stack = []
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return  F.softmax(out, dim=-1)



    def fwd(self, x, edge_index, de=None, epsilon=None):
        edge_weight = torch.ones(edge_index.shape[1]).type(torch.float).to(edge_index.device)
        if de is not None:
            if type(de) == int:
                de = [de]
            for e in de:
                edge_weight[e]=epsilon
                edl, edr = edge_index[0,e], edge_index[1,e]
                rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
                edge_weight[rev_de]=epsilon

        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return F.softmax(out, dim=-1)

    def fwd_base(self, x, edge_index):

        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index,)))
        for conv, ba, relu in zip(self.convs,self.bas,self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index)))
            #stack.append(post_conv)
        #print(stack)
        out_readout = self.readout(post_conv)

        out = self.ffn(out_readout)
        return out

    def embedding(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        #x, edge_index, batch = self.arguments_read(*args, **kwargs)
        #x, edge_index = args
        post_conv = self.relu1(self.ba1(self.conv1(x, edge_index, edge_weight)))
        for conv, ba, relu in zip(self.convs,self.bas, self.relus):
            post_conv = relu(ba(conv(post_conv, edge_index, edge_weight)))
        return post_conv

    def reset_parameters(self):

        init.xavier_uniform_(self.conv1.lin.weight)
        for conv in self.convs:
            init.xavier_uniform_(conv.lin.weight)
        init.xavier_uniform_(self.ffn[-1].weight)


    def __repr__(self):
        return self.__class__.__name__

class GNNPool(nn.Module):
    def __init__(self):
        super().__init__()

class IdenticalPool(GNNPool):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x



