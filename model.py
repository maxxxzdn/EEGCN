import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from utils import batch_dense_to_sparse

### GENERAL ARCHITECTURES ###


class ProjectorProcessorClassifier0(nn.Module):
    """
    An Projector-Processir-Classifier architecture. Base for this model.
    """

    def __init__(self, projector, processor, classifier):
        super(ProjectorProcessorClassifier0, self).__init__()
        self.projector = projector
        self.processor = processor
        self.classifier = classifier

    def forward(self, x, edge_index, batch_size):
        "Calculate latent representation for a graph and classify it."
        return self.classify(self.aggregate(self.process(
            self.project(x), edge_index), batch_size)), torch.zeros(1)

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, edge_index):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, edge_index)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation."
        return x.view(batch_size, -1)

    def classify(self, x):
        "Classify graph-level representation."
        return self.classifier(x)


class ProjectorProcessorClassifier1(nn.Module):
    """
    An Projector-Processir-Classifier architecture. Base for this model.
    """

    def __init__(self, projector, processor, classifier):
        super(ProjectorProcessorClassifier1, self).__init__()
        self.projector = projector
        self.processor = processor
        self.classifier = classifier

    def forward(self, x, edge_index, batch_size):
        "Calculate latent representation for a graph and classify it."
        x = torch.cat([self.process(x, edge_index), self.project(x)], dim=-1)
        return self.classify(self.aggregate(x, batch_size)), torch.zeros(1)

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, edge_index):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, edge_index)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation."
        return x.view(batch_size, -1)

    def classify(self, x):
        "Classify graph-level representation."
        return self.classifier(x)


class ProjectorProcessorClassifier2(nn.Module):
    """
    An Projector-Processir-Classifier architecture. Base for this model.
    """

    def __init__(self, projector, processor, classifier, attention, mask=None):
        super(ProjectorProcessorClassifier2, self).__init__()
        self.projector = projector
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask

    def forward(self, x, edge_index, batch_size):
        "Calculate latent representation for a graph and classify it."
        adj = self.calculate_adj(self.project(x), self.mask)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        x = self.process(x, edge_index, edge_weight)
        return self.classify(self.aggregate(x, batch_size)), adj

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, edge_index, edge_weight):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, edge_index, edge_weight)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation."
        return x.view(batch_size, -1)

    def calculate_adj(self, x, mask):
        "Calculate adjacency matrix based on latent representation."
        return self.attention(x, mask)

    def classify(self, x):
        "Classify graph-level representation."
        return self.classifier(x)


class ProjectorProcessorClassifier3(nn.Module):
    """
    An Projector-Processir-Classifier architecture. Base for this model.
    """

    def __init__(self, projector, processor, classifier,
                 attention, n_nodes, mask=None):
        super(ProjectorProcessorClassifier3, self).__init__()
        self.projector = projector
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask
        self.n_nodes = n_nodes

    def forward(self, x, edge_index, batch_size):
        "Calculate latent representation for a graph and classify it."
        adj = self.calculate_adj(self.project(x), self.mask)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        x = self.process(x, edge_index, edge_weight)
        return self.classify(self.aggregate(self.apply_node_mask(
            x, adj.sum(-1) + adj.sum(-2), batch_size), batch_size)), adj

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, edge_index, edge_weight):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, edge_index, edge_weight)

    def calculate_adj(self, x, mask=None):
        "Calculate adjacency matrix based on latent representation."
        return self.attention(x, mask)

    def apply_node_mask(self, x, node_weight, batch_size):
        "Apply node mask to feature matrix."
        return (node_weight.unsqueeze(-1) *
                x.view(batch_size, self.n_nodes, -1)).view(x.shape)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation with sum pooling."
        return x.view(batch_size, self.n_nodes, -1).sum(1)

    def classify(self, x):
        "Classify graph-level representation"
        return self.classifier(x)


class ProjectorProcessorClassifier4(nn.Module):
    "An Projector-Processir-Classifier architecture. Base for this model."

    def __init__(self, projector, processor, classifier,
                 attention, n_nodes, mask=None):
        super(ProjectorProcessorClassifier4, self).__init__()
        self.projector = projector
        self.processor = processor
        self.classifier = classifier
        self.attention = attention
        self.mask = mask
        self.n_nodes = n_nodes

    def forward(self, x, edge_index, batch_size):
        "Calculate latent representation for a graph and classify it."
        adj = self.calculate_adj(self.project(x), self.mask)
        edge_index, edge_weight = batch_dense_to_sparse(adj)
        x = self.process(self.apply_node_mask(
            x, adj.sum(-1) + adj.sum(-2), batch_size), edge_index, edge_weight)
        return self.classify(self.aggregate(x, batch_size)), adj

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, edge_index, edge_weight):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, edge_index, edge_weight)

    def calculate_adj(self, x, mask=None):
        "Calculate adjacency matrix based on latent representation."
        return self.attention(x, mask)

    def apply_node_mask(self, x, node_weight, batch_size):
        "Apply node mask to feature matrix."
        return (node_weight.unsqueeze(-1) *
                x.view(batch_size, self.n_nodes, -1)).view(x.shape)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation with sum pooling."
        return x.view(batch_size, self.n_nodes, -1).sum(1)

    def classify(self, x):
        "Classify graph-level representation."
        return self.classifier(x)

### CLASSIFIER ###


class Classifier(nn.Module):
    "Define standard linear + softmax classification step."

    def __init__(self, num_features, n_classes):
        super(Classifier, self).__init__()
        self.proj = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

### ENCODER ###


class PositionwiseFeedForward(nn.Module):
    "Implements f(x) = max(0, x*W1 + b1)*W2 + b2."

    def __init__(self, in_features, out_features, inner_features, p_dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features, inner_features)
        self.w_2 = nn.Linear(inner_features, out_features)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class _Projector(nn.Module):
    "Deprecated operator implemented based on 1D CNNs."

    def __init__(self, num_features, n_convs, pooling, p_dropout):
        super(_Projector, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers += nn.Sequential(nn.Conv1d(1, num_features, 7),
                                     nn.ReLU(), pooling, nn.Dropout(p_dropout))
        for _ in range(n_convs - 2):
            self.layers += nn.Sequential(
                nn.Conv1d(num_features, num_features, 5, 1, 2),
                nn.ReLU(),
                nn.Dropout(p_dropout))
        self.layers += nn.Sequential(pooling,
                                     nn.Conv1d(num_features, num_features, 3))
        self.norm = LayerNorm(num_features)

    def forward(self, x):
        "Pass the input data through each layer in turn."
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x.squeeze())

### PROCESSOR ###


class ProcessingBlock(nn.Module):
    "Processor is made up of self-attn and feed forward."

    def __init__(self, in_channels, out_channels, aggr, p_dropout):
        super(ProcessingBlock, self).__init__()
        self.layer = GraphConv(in_channels, out_channels, aggr=aggr)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return self.dropout(self.activation(
            self.norm(self.layer(x, edge_index, edge_weight))))


class Processor(nn.Module):
    "Core processor is a stack of N layers."

    def __init__(self, num_features, hidden_channels, aggr, n_hops, p_dropout):
        super(Processor, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(
            ProcessingBlock(num_features, hidden_channels, aggr, p_dropout))
        for _ in range(n_hops - 2):
            self.layers.append(
                ProcessingBlock(hidden_channels, hidden_channels, aggr, p_dropout))
        self.layers.append(
            GraphConv(hidden_channels, hidden_channels, aggr=aggr))
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_weight=None):
        "Pass the input data through each layer in turn."
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return self.dropout(self.norm(x))

### Attention ###


class AdjacencyAttention(nn.Module):
    "Attention operator to calculate adjacency matrix from the data."

    def __init__(self, num_features: int,
                 n_nodes: int, p_dropout: float):
        super(AdjacencyAttention, self).__init__()
        self.n_nodes = n_nodes
        self.num_features = num_features
        self._W1 = nn.Parameter(torch.FloatTensor(1))
        self._W2 = nn.Parameter(torch.FloatTensor(num_features, 1))
        self._W3 = nn.Parameter(torch.FloatTensor(num_features))
        self._bs = nn.Parameter(torch.FloatTensor(1, n_nodes, n_nodes))
        self._Vs = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        "Making a forward pass of the spatial attention layer."
        x = x.view(-1, self.n_nodes, self.num_features, 1)
        LHS = self.dropout(torch.matmul(torch.matmul(x, self._W1), self._W2))
        RHS = self.dropout(torch.matmul(self._W3, x).transpose(-1, -2))
        s = torch.matmul(self._Vs, torch.tanh(torch.matmul(LHS, RHS) + self._bs))
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        return self.dropout(F.softmax(s.reshape(x.size(0), -1), dim=-1)).view(-1, self.n_nodes,self.n_nodes)

### UTILS ###


def make_model(n_convs,n_hops, num_features, hidden_channels, aggr, pooling, positions, threshold, n, p_dropout=0.33, n_nodes=61, n_classes=2, signal_len=200):
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
    if n == 0:
        model = ProjectorProcessorClassifier0(
            projector=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
            processor=Processor(num_features, hidden_channels, aggr, n_hops, p_dropout),
            classifier=Classifier(n_nodes * hidden_channels, n_classes))
    elif n == 1:
        model = ProjectorProcessorClassifier1(
            projector=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
            processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
            classifier=Classifier(n_nodes * (hidden_channels + num_features), n_classes))
    elif n == 2:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag=True)
        else:
            mask = None
        model = ProjectorProcessorClassifier2(
            projector=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
            processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
            classifier=Classifier(n_nodes * hidden_channels, n_classes),
            attention=AdjacencyAttention(num_features, n_nodes, p_dropout / 2),
            mask=mask)
    elif n == 3:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag=True)
        else:
            mask = None
        model = ProjectorProcessorClassifier3(
            projector=PositionwiseFeedForward(signal_len, num_features, 100, p_dropout),
            processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
            classifier=Classifier(hidden_channels, n_classes),
            attention=AdjacencyAttention(num_features, n_nodes, p_dropout / 2),
            mask=mask,
            n_nodes=n_nodes)

    elif n == 4:
        if positions is not None:
            mask = distance_mask(positions, threshold, skip_diag=True)
        else:
            mask = None
        model = ProjectorProcessorClassifier4(
            projector=PositionwiseFeedForward(
                signal_len, num_features, 100, p_dropout),
            processor=Processor(signal_len, hidden_channels, aggr, n_hops, p_dropout),
            classifier=Classifier(hidden_channels, n_classes),
            attention=AdjacencyAttention(num_features, n_nodes, p_dropout / 2),
            mask=mask,
            n_nodes=n_nodes)

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model


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


def distance_mask(positions, threshold=1., skip_diag=True):
    "Mask nodes that are located far away from each other."
    dist_mat = torch.cdist(positions, positions)
    mask = torch.zeros(positions.size(0), positions.size(0)).to(dist_mat.device)
    mask = mask.masked_fill(dist_mat > threshold, 0)
    mask = mask.masked_fill(dist_mat < threshold, 1.)
    if skip_diag:
        mask = mask.fill_diagonal_(0)
    return mask

##########################################################################
