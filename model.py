import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


class Encoder(torch.nn.Module):
    """Takes a signal and produces a latent vector.
    Args:
        inp_dim (int): length of the signal.
        enc_dim (int): size of the hidden layer.
        latent_dim (int): dimension of the latent space.
    Returns:
        Latent representation of a signal.
    """

    def __init__(self, inp_dim, enc_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(inp_dim, enc_dim)
        self.fc2 = nn.Linear(enc_dim, enc_dim)
        self.fc3 = nn.Linear(enc_dim, enc_dim)
        self.fc4 = nn.Linear(enc_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Decoder(torch.nn.Module):
    """Takes a latent vector and produces a signal.
    Args:
        latent_dim (int): dimension of the latent space.
        enc_dim (int): size of the hidden layer.
        inp_dim (int): length of the signal.
    Returns:
        Reconstruction of the signal from its latent space representation.
    """

    def __init__(self, latent_dim, enc_dim, inp_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, enc_dim)
        self.fc2 = nn.Linear(enc_dim, enc_dim)
        self.fc3 = nn.Linear(enc_dim, enc_dim)
        self.fc4 = nn.Linear(enc_dim, inp_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class AE(torch.nn.Module):
    """ Takes an image and produces an image reconstructed from its latent space representation.
    Args:
        inp_dim (int): length of the signal.
        enc_dim (int): size of the hidden layer.
        latent_dim (int): dimension of the latent space.
    Returns:
        Reconstruction of the signal from its latent space representation.
    """

    def __init__(self, inp_dim, enc_dim, latent_dim):
        super(AE, self).__init__()
        self.encoder = Encoder(inp_dim, enc_dim, latent_dim)
        self.decoder = Decoder(latent_dim, enc_dim, inp_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Processor(torch.nn.Module):
    """ Performs N message passing steps to gather information all over a graph.
    Args:
        hops (int): number of message passing steps
        feature_dim (int): length of a feature vector.
        latent_dim_proc (int): dimension of the latent space of graph networks.
        latent_dim (int): dimension of the latent space.
        activation (torch.nn.functional): activation function.
    Returns:
        Updated latent representation of the graph.
    """

    def __init__(self, hops, feature_dim, latent_dim_proc,
                 latent_dim, activation=F.relu):
        super(Processor, self).__init__()
        torch.manual_seed(12345)
        self.activation = activation
        self.processor = nn.ModuleList([])
        self.processor.append(
            GraphConv(
                feature_dim,
                latent_dim_proc))
        for _ in range(hops - 1):
            self.processor.append(
                GraphConv(
                    latent_dim_proc,
                    latent_dim_proc))

        self.processor.append(
            GraphConv(
                latent_dim_proc,
                latent_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for gconv in self.processor[:-1]:
            x = gconv(x, edge_index, edge_weight)
            x = self.activation(x)
        x = self.processor[-1](x, edge_index, edge_weight)
        return x


class Pipeline(torch.nn.Module):
    """ Encodes a graph in embedded space and gather information all over a graph.
    Args:
        hops (int): number of message passing steps
        feature_dim (int): length of a feature vector.
        inp_dim (int): length of the signal.
        enc_dim (int): size of the hidden layer.
        latent_dim (int): dimension of the latent space.
        latent_dim_proc (int): dimension of the latent space of graph networks.
        activation (torch.nn.functional): activation function.
    Returns:
        Updated latent representation of the graph.
    """

    def __init__(self, hops, feature_dim, inp_dim, enc_dim,
                 latent_dim, latent_dim_proc):
        super(Pipeline, self).__init__()
        torch.manual_seed(12345)
        self.hops = hops
        self.latent_dim = latent_dim
        self.latent_dim_proc = latent_dim_proc
        self.autoencoder = AE(inp_dim, enc_dim, latent_dim)
        self.encoder = self.autoencoder.encoder
        self.decoder = self.autoencoder.decoder
        self.processor = Processor(
            hops, feature_dim, latent_dim_proc, latent_dim)

    def forward(self, x, edge_index, u, batch, positions, edge_weight=None):
        x = self.encoder(x)
        # Concatenate positions and global features to a feature vector
        x = torch.cat([x, positions, u[batch]], 1)
        x = self.processor(x, edge_index, edge_weight)
        return x

    def get_autoencoder(self, autoencoder, feature_dim):
        """ Assigns new autoencoder for the model."""
        self.autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.latent_dim = autoencoder.latent_dim
        self.processor = Processor(
            self.hops,
            feature_dim,
            self.latent_dim,
            self.latent_dim_proc)
