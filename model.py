import copy
import torch
# torch.manual_seed(12345)
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import matmul
from torch_geometric.nn import GraphConv
from torch_geometric.nn.conv import MessagePassing
from utils import batch_dense_to_sparse


### GENERAL ARCHITECTURES ###

class ProjectorProcessorClassifierAE(nn.Module):
    """
    An Projector-Processor-Classifier architecture. Base for this model.
    """

    def __init__(self, projector, preprocessor, processor, classifier, attention, autoencoder, mask):
        super(ProjectorProcessorClassifierAE, self).__init__()
        self.projector = projector
        self.processor = processor
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.attention = attention
        self.autoencoder = autoencoder
        self.mask = mask

    def forward(self, x, batch_size):
        "Calculate latent representation for a graph and classify it."
        x_enc = self.encode(x)
        adj = self.calculate_adj(self.project(x_enc), self.mask)
        return self.decode(x_enc), self.classify(self.aggregate(self.process(self.preprocess(x_enc), adj), batch_size)), adj

    def encode(self, x):
        "Calculate latent representation from original data."
        return self.autoencoder.encoder(x)

    def decode(self, x):
        "Calculate original data from latent representation."
        return self.autoencoder.decoder(x)

    def project(self, x):
        "Project raw data to latent space."
        return self.projector(x)

    def process(self, x, adj):
        "Calculate node embeddings via learned message-passing."
        return self.processor(x, adj)

    def preprocess(self, x):
        "Project raw data to latent space."
        return self.preprocessor(x)

    def aggregate(self, x, batch_size):
        "Aggregate node fetures into graph-level representation."
        return x.view(batch_size, -1)

    def calculate_adj(self, x, mask):
        "Calculate adjacency matrix based on latent representation."
        return self.attention(x, mask)

    def classify(self, x):
        "Classify graph-level representation."
        return F.log_softmax(self.classifier(x), dim=-1)

### ENCODER ###


class Encoder(torch.nn.Module):
    """Takes an image and produces a latent vector."""

    def __init__(self, d_input, d_enc, d_latent, N, norm=True):
        super(Encoder, self).__init__()
        self.layers = clones(nn.Linear(d_enc, d_enc), N-2)
        self.layers.insert(0, nn.Linear(d_input, d_enc))
        self.layers.append(nn.Linear(d_enc, d_latent))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class ConvEncoder(torch.nn.Module):
    def __init__(self, d_latent, d_hidden, kernel_size):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size, 1, kernel_size//2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size, 1, kernel_size//2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size, 1, kernel_size//2)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(25*128, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_latent)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)

### DECODER ###


class Decoder(torch.nn.Module):
    """ Takes a latent vector and produces an image."""

    def __init__(self, d_input, d_enc, d_latent, N):
        super(Decoder, self).__init__()
        self.layers = clones(nn.Linear(d_enc, d_enc), N-2)
        self.layers.insert(0, nn.Linear(d_latent, d_enc))
        self.layers.append(nn.Linear(d_enc, d_input))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class ConvDecoder(torch.nn.Module):
    def __init__(self, d_latent, d_hidden, kernel_size):
        super(ConvDecoder, self).__init__()
        self.fc1 = nn.Linear(d_latent, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 25*128)
        self.t_conv1 = nn.Conv1d(128, 32, kernel_size, 1, kernel_size//2)
        self.t_conv2 = nn.Conv1d(32, 16, kernel_size, 1, kernel_size//2)
        self.t_conv3 = nn.Conv1d(16, 1, kernel_size, 1, kernel_size//2)
        self.ups = torch.nn.Upsample(scale_factor=(
            2), mode="linear", align_corners=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 25)
        x = F.relu(self.t_conv1(self.ups(x)))
        x = F.relu(self.t_conv2(self.ups(x)))
        x = F.relu(self.t_conv3(self.ups(x)))
        return x.squeeze()

### AUTOENCODER ###


class Autoencoder(torch.nn.Module):
    def __init__(self, d_input, d_enc, d_latent, N):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(d_input, d_enc, d_latent, N)
        self.decoder = Decoder(d_input, d_enc, d_latent, N)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def __repr__(self):
        return self.__class__.__name__


class ConvAutoencoder(torch.nn.Module):
    def __init__(self, d_latent, kernel_size, d_hidden=256):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(d_latent, d_hidden, kernel_size)
        self.decoder = ConvDecoder(d_latent, d_hidden, kernel_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def __repr__(self):
        return self.__class__.__name__

### PROCESSOR ###


class AdjConv(MessagePassing):
    def __init__(self, adj, n_hops, d_input, d_output, aggr, **kwargs):
        super(AdjConv, self).__init__(aggr=aggr, **kwargs)
        self.n_hops = n_hops
        self.layer = nn.Linear(d_input, d_output, bias=True)
        self.adjs = adj_power(adj, n_hops, False).unsqueeze(0)

    def forward(self, x, adj_weight):
        adj_weight = torch.mul(adj_power_batch(
            adj_weight, self.n_hops, True), self.adjs)
        edge_index_all, edge_weight_all = torch.empty(2, 0).to(
            x.device).long(), torch.empty(0).to(x.device)
        for i in range(self.n_hops):
            edge_index, edge_weight = batch_dense_to_sparse(adj_weight[:, i])
            edge_index_all = torch.cat([edge_index_all, edge_index], 1)
            edge_weight_all = torch.cat([edge_weight_all, edge_weight], 0)
        return self.layer(self.propagate(edge_index_all, x=x, edge_weight=edge_weight_all)) + x

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}, #hops: {}'.format(self.__class__.__name__, self.n_hops)


class Processor(nn.Module):
    "Core processor is a stack of N layers."

    def __init__(self, adj, n_hops, d_input, d_output, aggr, p_dropout):
        super(Processor, self).__init__()
        self.conv = AdjConv(adj, n_hops, d_input, d_output, aggr)
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(d_output)

    def forward(self, x, adj_weight):
        "Pass the input data through each layer in turn."
        return self.dropout(self.norm(self.conv(x, adj_weight)))

### ATTENTION ###


class AdjacencyAttention(nn.Module):
    "Attention operator to calculate adjacency matrix from the data."

    def __init__(self, d_input: int,
                 n_nodes: int, p_dropout: float):
        super(AdjacencyAttention, self).__init__()
        self.n_nodes = n_nodes
        self.d_input = d_input
        self._W1 = nn.Parameter(torch.FloatTensor(1))
        self._W2 = nn.Parameter(torch.FloatTensor(d_input, 1))
        self._W3 = nn.Parameter(torch.FloatTensor(d_input))
        self._bs = nn.Parameter(torch.FloatTensor(1, n_nodes, n_nodes))
        self._Vs = nn.Parameter(torch.FloatTensor(n_nodes, n_nodes))
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x, mask=None):
        "Making a forward pass of the spatial attention layer."
        x = x.view(-1, self.n_nodes, self.d_input, 1)
        LHS = self.dropout(torch.matmul(torch.matmul(x, self._W1), self._W2))
        RHS = self.dropout(torch.matmul(self._W3, x).transpose(-1, -2))
        s = torch.matmul(self._Vs, torch.tanh(
            torch.matmul(LHS, RHS) + self._bs))
        if mask is not None:
            s = s.masked_fill(mask == 0, -1e9)
        return self.dropout(F.softmax(s.reshape(x.size(0), -1), dim=-1)).view(-1, self.n_nodes, self.n_nodes)

### UTILS ###


def make_model(n_hops, num_features, aggr, positions, threshold, autoencoder, kernel_size=11, N=4, n_nodes=61, n_classes=2, signal_len=200):
    "Helper: Construct a model from hyperparameters."
    assert aggr in ["max", "add", "mean"]

    if positions is not None:
        mask = distance_mask(positions, threshold, skip_diag=True)
        print(str(int(mask.sum().item())) + " edges considered.")
    else:
        raise NotImplementedError
    
    if autoencoder == 'ffn':
        autoencoder = Autoencoder(signal_len, signal_len//2, num_features, N)
    elif autoencoder == 'conv':
        autoencoder = ConvAutoencoder(num_features, kernel_size)
        

    model = ProjectorProcessorClassifierAE(
        projector=Projector(num_features, num_features, p_dropout=0.2),
        preprocessor=Projector(num_features, num_features, p_dropout=0.4),
        processor=Processor(mask, n_hops, num_features,
                            num_features, aggr, p_dropout=0.4),
        classifier=Projector(n_nodes * num_features,
                             n_classes, p_dropout=0.0, classify=True),
        attention=AdjacencyAttention(num_features, n_nodes, p_dropout=0.2),
        autoencoder=autoencoder,
        mask=mask)

    print(str(model.processor.conv) + " is the model's processor.")
    print(str(model.autoencoder) + " is the model's autoencoder.")

    for parameter in model.named_parameters():
        name, parameter = parameter
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
        if name == 'attention._bs':
            nn.init.zeros_(parameter)
            
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

    def backward(self, y, mean, std):
        return (y - self.b_2)*(std + self.eps)/self.a_2 + mean


def distance_mask(positions, threshold=1., skip_diag=True):
    "Mask nodes that are located far away from each other."
    dist_mat = torch.cdist(positions, positions)
    mask = torch.zeros(positions.size(
        0), positions.size(0)).to(dist_mat.device)
    mask = mask.masked_fill(dist_mat > threshold, 0)
    mask = mask.masked_fill(dist_mat < threshold, 1.)
    if skip_diag:
        mask = mask.fill_diagonal_(0)
    return mask


def adj_power(adj, n_hops, weight):
    assert adj.size(-1) == adj.size(-2)
    out = torch.empty(n_hops, adj.size(-1), adj.size(-1)).to(adj.device)
    for i in range(n_hops):
        out[i] = adj if weight else (adj == 1.).double()
        adj = torch.matmul(adj, adj)
    return out


def adj_power_batch(adj, n_hops, weight):
    assert len(adj.size()) > 2
    out = torch.empty(adj.size(0), n_hops, adj.size(-1),
                      adj.size(-1)).to(adj.device)
    for i in range(adj.size(0)):
        out[i] = adj_power(adj[i], n_hops, weight)
    return out


def clones(module, N, shared=False):
    "Produce N identical layers."
    if shared:
        return nn.ModuleList(N*[module])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class FeedForwardNetwork(nn.Module):
    "Implements f(x) = max(0, x*W1 + b1)*W2 + b2."

    def __init__(self, d_input, d_output, d_ff, p_dropout):
        super(FeedForwardNetwork, self).__init__()
        self.w_1 = nn.Linear(d_input, d_ff)
        self.w_2 = nn.Linear(d_ff, d_output)
        self.dropout = nn.Dropout(p_dropout)
        self.norm = LayerNorm(d_output)

    def forward(self, x):
        return self.norm(self.w_2(self.dropout(F.relu(self.w_1(x)))))


class Projector(nn.Module):
    "Defines linear projection from one space to another."

    def __init__(self, d_input, d_output, p_dropout, classify=False, bias=True):
        super(Projector, self).__init__()
        self.proj = nn.Linear(d_input, d_output, bias=bias)
        self.classify = classify
        if not classify:
            self.norm = LayerNorm(d_output)
            self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        "Project the input data to another space."
        return self.dropout(self.norm(self.proj(x))) if not self.classify else self.proj(x)

    def backward(self, y, mean, std):
        "Project the output data to the original space."
        bias = self.proj.bias if self.proj.bias is not None else 0
        return torch.matmul(self.norm.backward(y, mean, std) - bias, torch.pinverse(self.proj.weight.T))
