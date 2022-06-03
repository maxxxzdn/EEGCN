import torch
import copy
import numpy as np
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

            
class Encoder(torch.nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.activation = get_activation(args)
        self.pooling = get_pooling(args)[0]
        self.ks = args.kernel_size
        
        self.convs = nn.ModuleList([])
        self.convs.append(Block(1, args.d_latent, self.ks, 1, self.ks//2, args.norm_enc, False))
        self.convs += clones(Block(args.d_latent, args.d_latent, 5, 1, 2, args.norm_enc, True), args.n_cnn-2)
        self.convs.append(Block(args.d_latent, 8, 5, 1, 2, args.norm_enc, False))       
        self.dropout = nn.Dropout(args.p_dropout)
        
    def forward(self, x):
        z = x.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            z = conv(z)
            if i == 0 or i == (len(self.convs)-1):
                z = self.pooling(z)
        return self.dropout(z)
    
class Processor(torch.nn.Module):
    def __init__(self, args):
        super(Processor, self).__init__()
        self.activation = get_activation(args)
        self.gconvs = nn.ModuleList([])
        self.gconvs.append(GBlock(32, args.d_hidden, args.norm_proc, False))
        self.gconvs += clones(GBlock(args.d_hidden, args.d_hidden, args.norm_proc, True), args.n_mp-2)
        self.gconvs.append(GBlock(args.d_hidden, args.d_hidden, args.norm_proc, False))
        self.dropout = nn.Dropout(args.p_dropout)
        
    def forward(self, z, edge_index, edge_weight, batch):
        for gconv in self.gconvs:
            z = gconv(z, edge_index, edge_weight, batch)
        return self.dropout(z)

class Classifier(torch.nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.activation = get_activation(args)
        self.mlp = nn.Sequential(
                       nn.Linear(args.d_hidden, args.d_hidden//2),
                       nn.LayerNorm(args.d_hidden//2),
                       nn.ReLU())
        self.linear = nn.Linear(args.d_hidden//2, 3)
             
    def forward(self, h):
        h = self.mlp(h)
        return self.linear(h)
                   
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__() 
        # 1D Convolutions to extract features from a signal                    
        self.encoder = Encoder(args) if args.n_cnn > 0 else Identity()
        # Graph Convolution Networks
        self.processor = Processor(args) if args.n_mp > 0 else Identity()
        if args.aggregate == 'eq':
            self.aggregator = EquilibriumAgg(args.d_hidden, t = 10)
        else:
            self.aggregator = globPool(args.aggregate)
        # MLP to make a prediction  
        self.classifier = Classifier(args)

    def forward(self, x, edge_index, edge_weight = None, batch = None):
        z = self.encoder(x).view(x.shape[0], -1)
        h = self.processor(z, edge_index, edge_weight, batch)
        h, grads = self.aggregator(h, batch)
        y = self.classifier(h)
        if self.training:
            return torch.sigmoid(y), grads
        else:
            return torch.sigmoid(y)

def get_activation(args):
    if args.activation == 'leaky_relu':
        return F.leaky_relu
    elif args.activation == 'relu':
        return F.relu
    elif args.activation == 'tanh':
        return F.tanh
    else: 
        raise NotImplementedError

def get_pooling(args):
    if args.pooling == "max":
        return nn.MaxPool1d(kernel_size = 5), gnn.global_max_pool
    elif args.pooling == "avg":
        return nn.AvgPool1d(kernel_size = 5), gnn.global_mean_pool
    else:
        raise NotImplementedError
        
def clones(module, N, shared=False):
    "Produce N identical layers."
    if shared:
        return nn.ModuleList(N*[module])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
class LayerNorm(nn.Module):
    "Construct a layer norm module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

    def backward(self, y, mean, std):
        return (y - self.b_2)*(std + self.eps)/self.a_2 + mean
    
class Concatenate(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim
    def forward(self, input):
        return input.view(-1, 61*self.in_dim)
    
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm, skipcon):
        super(Block, self).__init__()
        self.skipcon = skipcon
        if norm:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = Identity
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        if self.skipcon:
            out += identity
            out = self.relu(out)
        return out
    
class gBatchNorm(gnn.BatchNorm):
    def __init__(self, in_channels):
        super().__init__(in_channels)
        
    def forward(self, x, batch):
        return self.module(x)

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm, skipcon):
        super(GBlock, self).__init__()
        self.skipcon = skipcon
        if norm == 'batch':
            norm_layer = gBatchNorm
        elif norm == 'graph':
            norm_layer = gnn.GraphNorm
        elif norm == 'layer':
            norm_layer = gnn.LayerNorm
        else:
            norm_layer = Identity
        self.gconv = gnn.GraphConv(in_channels, out_channels)
        self.norm = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, edge_index, edge_weight, batch):
        identity = x
        out = self.gconv(x, edge_index, edge_weight)
        out = self.norm(out, batch)
        out = self.relu(out)
        if self.skipcon:
            out += identity
            out = self.relu(out)
        return out    
    
class Identity(torch.nn.Module):
    def __init__(self, *args):
        super().__init__()
        pass
    def forward(*args):
        return args[1]
    
class globPool():
    def __init__(self, aggr):
        if aggr == 'max':
            self.aggr = gnn.global_max_pool
        elif aggr == 'mean':
            self.aggr = gnn.global_mean_pool
    def __call__(self, x, batch):
        return self.aggr(x, batch), torch.zeros(1).to(x.device)
