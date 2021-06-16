import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GCNConv
from utils import batch_dense_to_sparse
from torch_geometric.utils import remove_self_loops

### GENERAL ARCHITECTURES ###
class EncoderProcessorClassifier0(nn.Module):
    """
    An Encoder-Processir-Classifier architecture. Base for this model.
    """
    def __init__(self, encoder, processor, classifier):
        super(EncoderProcessorClassifier0, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.classifier = classifier

    def forward(self, x, edge_index, batch):
        "Calculate latent representation for a graph and classify it"
        return self.classify(self.process(
            self.encode(x), edge_index))

    def encode(self, x):
        return self.encoder(x)

    def process(self, x, edge_index):
        return self.processor(x, edge_index).reshape((batch.max().item() + 1), -1)

    def classify(self, x):
        return self.classifier(x)
    
class EncoderProcessorClassifier1(nn.Module):
    """
    An Encoder-Processir-Classifier architecture. Base for this model.
    """
    def __init__(self, encoder, processor, classifier):
        super(EncoderProcessorClassifier1, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.classifier = classifier

    def forward(self, x, edge_index, batch):
        "Calculate latent representation for a graph and classify it"
        x = torch.cat([self.process(x, edge_index), self.encode(x)], dim=-1)
        x = x.reshape((batch.max().item() + 1), -1)
        return self.classify(x)

    def encode(self, x):
        return self.encoder(x)

    def process(self, x, edge_index):
        return self.processor(x, edge_index)

    def classify(self, x):
        return self.classifier(x)
    
class EncoderProcessorClassifier2(nn.Module):
    """
    An Encoder-Processir-Classifier architecture. Base for this model.
    """
    def __init__(self, encoder, processor, classifier, attention, mask=None):
        super(EncoderProcessorClassifier2, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask

    def forward(self, x, edge_index, batch):
        "Calculate latent representation for a graph and classify it"
        adj = self.calculate_adj(self.encode(x), self.mask)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        x = self.process(x, edge_index, edge_weight).reshape((batch.max().item() + 1), -1)
        return self.classify(x), adj

    def encode(self, x):
        return self.encoder(x)

    def process(self, x, edge_index, edge_weight):
        return self.processor(x, edge_index, edge_weight)
    
    def calculate_adj(self, x, mask=None):
        return self.attention(x, mask)

    def classify(self, x):
        return self.classifier(x) 
    
class EncoderProcessorClassifier3(nn.Module):
    """
    An Encoder-Processir-Classifier architecture. Base for this model.
    """
    def __init__(self, encoder, processor, classifier, attention, mask=None):
        super(EncoderProcessorClassifier3, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask

    def forward(self, x, edge_index, batch):
        "Calculate latent representation for a graph and classify it"
        adj = self.calculate_adj(self.encode(x), self.mask)
        node_weight = adj.sum(-1) + adj.sum(-2) #adj.diagonal(dim1 = -2, dim2 = -1)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        edge_index, edge_weight =  remove_self_loops(edge_index, edge_weight)
        x = self.process(x, edge_index, edge_weight)
        x = self.adj_pooling(x, node_weight, batch.max().item() + 1)
        return self.classify(x), adj

    def encode(self, x):
        return self.encoder(x)

    def process(self, x, edge_index, edge_weight):
        return self.processor(x, edge_index, edge_weight)
    
    def calculate_adj(self, x, mask=None):
        return self.attention(x, mask)
    
    def adj_pooling(self, x, node_weight, batch_size=32, n_nodes=61):
        return (node_weight.reshape(batch_size, n_nodes, 1)*x.reshape(batch_size, n_nodes, -1)).sum(1)

    def classify(self, x):
        return self.classifier(x)
    
class EncoderProcessorClassifier4(nn.Module):
    """
    An Encoder-Processir-Classifier architecture. Base for this model.
    """
    def __init__(self, encoder, processor, classifier, attention, mask=None):
        super(EncoderProcessorClassifier4, self).__init__()
        self.encoder = encoder
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask

    def forward(self, x, edge_index, batch):
        "Calculate latent representation for a graph and classify it"
        adj = self.calculate_adj(self.encode(x), self.mask)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        edge_index, edge_weight =  remove_self_loops(edge_index, edge_weight)
        x = self.process(self.adj_pooling(x, adj), edge_index, edge_weight)
        x = x.reshape(batch.max().item() + 1, 61, -1).sum(1).reshape(batch.max().item() + 1, -1)
        return self.classify(x), adj

    def encode(self, x):
        return self.encoder(x)

    def process(self, x, edge_index, edge_weight):
        return self.processor(x, edge_index, edge_weight)
    
    def calculate_adj(self, x, mask=None):
        return self.attention(x, mask)
    
    def adj_pooling(self, x, adj, n_nodes = 61, threshold = 0.01):
        mask = (adj.sum(-1) + adj.sum(-2)).reshape(-1,1) #adj.masked_fill(adj < threshold, 0.)
        #edge_index, _ = batch_dense_to_sparse(adj)
        #mask = torch.zeros_like(x)
        #mask[torch.unique(edge_index.reshape(-1))] = 1.
        return torch.mul(x, mask)
        
    def classify(self, x):
        return self.classifier(x)
    
### CLASSIFIER ###
class Classifier(nn.Module):
    "Define standard linear + softmax classification step."
    def __init__(self, d_model, n_classes):
        super(Classifier, self).__init__()
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

### ENCODER ###
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, in_features, out_features, inner_features, p_dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features, inner_features)
        self.w_2 = nn.Linear(inner_features, out_features)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self, num_features, n_convs, pooling, p_dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers += nn.Sequential(nn.Conv1d(1, num_features, 7),
                                  nn.ReLU(), pooling, nn.Dropout(p_dropout))
        for _ in range(n_convs - 2):
            self.layers += nn.Sequential(nn.Conv1d(num_features,
                                             num_features, 5, 1, 2), nn.ReLU(), nn.Dropout(p_dropout))
        self.layers += nn.Sequential(pooling,
                                  nn.Conv1d(num_features, num_features, 3))
        self.norm = LayerNorm(num_features)

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x.squeeze())

### PROCESSOR ###
class ProcessingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, aggr, p_dropout):
        super(ProcessingBlock, self).__init__()
        self.layer = GraphConv(in_channels, out_channels, aggr=aggr) #GCNConv(in_channels, out_channels, add_self_loops = False, aggr=aggr)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return x + self.dropout(self.activation(self.norm(self.layer(x, edge_index, edge_weight))))

class Processor(nn.Module):
    """Obtains node embeddings."""
    def __init__(self, num_features, hidden_channels, aggr, n_hops, p_dropout):
        super(Processor, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(
            ProcessingBlock(
                num_features,
                hidden_channels,
                aggr,
                p_dropout))
        for _ in range(n_hops - 2):
            self.layers.append(
                ProcessingBlock(
                    hidden_channels,
                    hidden_channels,
                    aggr,
                    p_dropout))
        self.layers.append(
            GraphConv(
                hidden_channels,
                hidden_channels,
                aggr=aggr))
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return self.dropout(self.norm(x)) 

### Attention ###
class GraphAttention(nn.Module):
    """
    Args:
        in_channels (int): Number of input features.
        num_of_vertices (int): Number of vertices in the graph.
        num_of_timesteps (int): Number of time lags.
    """
    def __init__(self, in_channels: int, num_of_vertices: int, num_of_timesteps: int, p_dropout: float):
        super(GraphAttention, self).__init__()
        
        self.num_of_vertices = num_of_vertices
        self.in_channels = in_channels
        self._W1 = nn.Parameter(torch.FloatTensor(num_of_timesteps))
        self._W2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_timesteps))
        self._W3 = nn.Parameter(torch.FloatTensor(in_channels))
        self._bs = nn.Parameter(torch.FloatTensor(1, num_of_vertices, num_of_vertices))
        self._Vs = nn.Parameter(torch.FloatTensor(num_of_vertices, num_of_vertices))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, X: torch.FloatTensor, mask: torch.FloatTensor=None) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention layer.
        
        Arg types:
            * **X** (PyTorch FloatTensor) - Node features for T time periods, with shape (B, N_nodes, F_in).

        Return types:
            * **S** (PyTorch FloatTensor) - Spatial attention score matrices, with shape (B, N_nodes, N_nodes).
        """
        X = X.reshape(-1,self.num_of_vertices,self.in_channels,1)
        batch_size = X.size(0)
        LHS = torch.matmul(torch.matmul(X, self._W1), self._W2)
        LHS = self.dropout(LHS)
        RHS = torch.matmul(self._W3, X).transpose(-1, -2)
        RHS = self.dropout(RHS)
        S = torch.matmul(self._Vs, torch.sigmoid(torch.matmul(LHS, RHS) + self._bs))
        if mask is not None:
            S = S.masked_fill(mask==0, -1e9)
        S = S.reshape(batch_size, -1)
        S = F.softmax(S, dim = 1)
        S = S.reshape(batch_size, self.num_of_vertices, self.num_of_vertices)
        return self.dropout(S)
    
class AttentionPooling(nn.Module):
    def __init__(self, in_features: int, inner_features: int, num_of_vertices: int, p_dropout: float):
        super(AttentionPooling, self).__init__()   
        self.num_of_vertices = num_of_vertices
        self.in_features = in_features
        self.proj = PositionwiseFeedForward(in_features, 1, inner_features, p_dropout)

    def forward(self, X, Z):
        batch_size = X.size(1)//self.num_of_vertices
        X = X.reshape(batch_size, self.num_of_vertices, -1)
        Z = Z.reshape(batch_size, self.num_of_vertices, -1)
        return (X*F.log_softmax(self.proj(Z),1)).sum(1)
    
### UTILS ###
def make_model(n_convs, n_hops, num_features, hidden_channels,
               aggr, pooling, positions, threshold, p_dropout=0.1, n_nodes=61, n_classes=2, n=4, signal_len=200):
    "Helper: Construct a model from hyperparameters."
    assert n_convs >= 2
    assert n_hops >= 1
    assert aggr in ["max", "add"]
    if pooling == "max":
        pooling = nn.MaxPool1d(kernel_size=7)
    elif pooling == "avg":
        pooling = nn.AvgPool1d(kernel_size=7)
    else:
        raise NotImplementedError
    if n==0:
        model = EncoderProcessorClassifier0(encoder=Encoder(num_features, n_convs, pooling, p_dropout),
                                       processor=Processor(num_features, hidden_channels, aggr, n_hops, p_dropout),
                                       classifier=Classifier(n_nodes * hidden_channels, n_classes))
    elif n==1:
        model = EncoderProcessorClassifier1(encoder=Encoder(num_features, n_convs, pooling, p_dropout),
                                       processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
                                       classifier=Classifier(n_nodes * (hidden_channels+num_features), n_classes))
    elif n==2:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag = True)
        else:
            mask = None
        model = EncoderProcessorClassifier2(
                                       #encoder=Encoder(num_features, n_convs, pooling, p_dropout),
                                       encoder=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
                                       processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
                                       classifier=Classifier(n_nodes * hidden_channels, n_classes),
                                       attention=GraphAttention(num_features, n_nodes, 1, p_dropout/2),
                                       mask=mask) 
    elif n==3:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag = True)
        else:
            mask = None
        model = EncoderProcessorClassifier3(
                                       #encoder=Encoder(num_features, n_convs, pooling, p_dropout),
                                       encoder=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
                                       processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
                                       classifier=Classifier(hidden_channels, n_classes),
                                       attention=GraphAttention(num_features, n_nodes, 1, p_dropout/2),
                                       mask=mask) 
        
    elif n==4:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag = True)
        else:
            mask = None
        model = EncoderProcessorClassifier4(
                                       #encoder=Encoder(num_features, n_convs, pooling, p_dropout),
                                       encoder=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
                                       processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
                                       classifier=Classifier(hidden_channels, n_classes),
                                       attention=GraphAttention(num_features, n_nodes, 1, p_dropout/2),
                                       mask=mask) 

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
def distance_mask(positions, threshold=1., skip_diag=True):
    d = torch.cdist(positions,positions)
    mask = torch.zeros(positions.size(0),positions.size(0)).to(d.device)
    mask = mask.masked_fill(d > threshold, 0)
    mask = mask.masked_fill(d < threshold, 1.)
    if skip_diag:
        mask = mask.fill_diagonal_(0)
    return mask

###################################################################################
