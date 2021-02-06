import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList, Sequential, Softmax, ReLU, Dropout, BatchNorm1d


class Encoder(torch.nn.Module):
    def __init__(self, num_features, convs=4):
        super(Encoder, self).__init__()
        torch.manual_seed(12345)

        self.num_features = num_features
        self.convs = convs
        self.activation = F.relu
        self.pooling = AvgPool1d(kernel_size=7)

        self.encoder = ModuleList([])
        self.encoder.append(
            Conv1d(
                in_channels=1,
                out_channels=self.num_features,
                kernel_size=7))
        for _ in range(self.convs - 2):
            self.encoder.append(
                Conv1d(
                    in_channels=self.num_features,
                    out_channels=self.num_features,
                    kernel_size=5,
                    padding=2))
        self.encoder.append(
            Conv1d(
                in_channels=self.num_features,
                out_channels=self.num_features,
                kernel_size=3))

    def forward(self, x):

        rows = x.shape[0]  # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1]  # length of a signal
        x = x.reshape(rows, 1, cols)  # reshape to 1 channel

        # 1. Extract features from time series
        x = self.encoder[0](x)
        x = self.activation(x)
        x = self.pooling(x)
        x = F.dropout(x, p=0.25, training=self.training)

        for conv1d in self.encoder[1:-1]:
            x = conv1d(x)
            x = self.activation(x)
            x = F.dropout(x, p=0.25, training=self.training)

        x = self.pooling(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.encoder[-1](x)
        x = x.squeeze()

        return x


class Classifier(torch.nn.Module):
    def __init__(self, num_features, hidden_channels,
                 n_classes=3, hops=8, layers=6, convs=4):
        super(Classifier, self).__init__()
        torch.manual_seed(12345)

        # check if architecture requested is meaningful
        assert hops > 0
        assert layers > 0
        assert convs > 1

        self.hops = hops
        self.layers = layers
        self.convs = convs
        self.n_classes = n_classes
        self.num_features = num_features
        self.graph_operator = GraphConv
        self.activation = F.relu
        self.pooling = AvgPool1d(kernel_size=7)
        self.graphPooling = global_mean_pool

        # 1D Convolutions to extract features from a signal
        self.encoder = Encoder(self.num_features, self.convs)

        # Graph Convolution Networks
        self.gconvs = ModuleList([])
        self.gconvs.append(GraphConv(self.num_features, hidden_channels))
        for _ in range(self.hops - 1):
            self.gconvs.append(GraphConv(hidden_channels, hidden_channels))

        # Linear layers to make a prediction
        self.linlayers = ModuleList([])
        for _ in range(self.layers - 1):
            self.linlayers.append(Linear(hidden_channels, hidden_channels))
        self.linlayers.append(Linear(hidden_channels, self.n_classes))

    def forward(self, x, edge_index, batch, edge_weight=None):

        x = self.encoder(x)
        # 2. Obtain node embeddings; x.shape = [61,node_features]
        for gconv in self.gconvs[:-1]:
            x = gconv(x, edge_index, edge_weight)  # [61, hidden_channels]
            x = self.activation(x)
            x = F.dropout(x, p=0.25, training=self.training)
        x = self.gconvs[-1](x, edge_index, edge_weight)

        # 3. Readout layer
        x = self.graphPooling(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        for linlayer in self.linlayers[:-1]:
            x = linlayer(x)
            x = self.activation(x)
        x = self.linlayers[-1](x)

        return F.softmax(x, -1)


"""
class Encoder(torch.nn.Module):
    def __init__(self, num_features):
        super(Encoder, self).__init__()
        torch.manual_seed(12345)

        assert num_features >= 6

        self.activation = ReLU()
        self.pooling = AvgPool1d(kernel_size = 7)

        # check if architecture requested is meaningful
        self.num_features = (num_features//3)*3
        channels = self.num_features//3

        # 1D Convolutions to extract features from a signal

        self.hf_encoder = Sequential(
                          Conv1d(1, channels, 3, 2, 1),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 3, 2, 1),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 3, 2, 1),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 3, 2, 1),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 3, 2, 1),
                          self.pooling
        )

        self.mf_encoder = Sequential(
                          Conv1d(1, channels, 7, 2, 3),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 7, 2, 3),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 7, 2, 3),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 7, 2, 3),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 7, 2, 3),
                          self.pooling
        )

        self.lf_encoder = Sequential(
                          Conv1d(1, channels, 11, 2, 5),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 11, 2, 5),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 11, 2, 5),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 11, 2, 5),
                          self.activation,
                          Dropout(0.25),
                          Conv1d(channels, channels, 11, 2, 5),
                          self.pooling
        )

    def forward(self, x):

        rows = x.shape[0] # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1] # length of a signal
        x = x.reshape(rows,1,cols) # reshape to 1 channel

        # 1. Extract features from time series
        x1 = self.hf_encoder(x).squeeze()
        x2 = self.mf_encoder(x).squeeze()
        x3 = self.lf_encoder(x).squeeze()

        x = torch.cat([x1,x2,x3],1)

        return x
"""


class Decoder(torch.nn.Module):
    def __init__(self, num_features, hidden_size, out_size):
        super(Decoder, self).__init__()
        torch.manual_seed(12345)

        self.activation = ReLU()

        # 1D Convolutions to extract features from a signal
        # self.classifier = Sequential(Linear(num_features_global, hidden_size),
        #                             self.activation,
        #                             Linear(hidden_size, hidden_size),
        #                             self.activation,
        #                             Linear(hidden_size, 3),
        #                             Softmax(dim=-1)
        #                             )
        self.decoder = Sequential(Linear(num_features, hidden_size),
                                  self.activation,
                                  Linear(hidden_size, hidden_size),
                                  self.activation,
                                  Linear(hidden_size, out_size)
                                  )

    def forward(self, x):  # , u):

        x = self.decoder(x)
        #u = self.classifier(u)

        return x  # ,u


class Processor(torch.nn.Module):
    def __init__(self, hops, num_features, hidden_channels):
        super(Processor, self).__init__()
        torch.manual_seed(12345)

        self.hops = hops
        self.num_features = num_features + 3
        self.hidden_channels = hidden_channels

        self.processor = ModuleList([])
        self.processor.append(
            GraphConv(
                self.num_features,
                self.hidden_channels))
        for _ in range(self.hops - 1):
            self.processor.append(
                GraphConv(
                    self.hidden_channels,
                    self.hidden_channels))

    def forward(self, x, edge_index, u, batch):

        x = torch.cat([x, u[batch]], dim=1)

        for gconv in self.processor:
            x = gconv(x, edge_index)

        return x


class Pipeline(torch.nn.Module):
    def __init__(self, hops, num_features, hidden_size, out_size):
        super(Pipeline, self).__init__()
        torch.manual_seed(12345)

        self.classifier = Classifier(num_features, hidden_size)
        self.encoder = self.classifier.encoder.eval()
        self.processor = Processor(hops, num_features, hidden_size)
        self.decoder = Decoder(hidden_size, hidden_size, out_size)

    def forward(self, x, edge_index, u, batch):

        x = self.encoder(x)
        x = self.processor(x, edge_index, u, batch)
        x = self.decoder(x)

        return x
