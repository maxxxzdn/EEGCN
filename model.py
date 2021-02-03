import torch
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList
from torch_geometric.nn import MetaLayer
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList, Sequential, Softmax, ReLU, Dropout, BatchNorm1d, ReLU
from torch_scatter import scatter_mean


class NodeModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, u_size):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(
            Linear(
                num_features, hidden_size), ReLU(), Linear(
                hidden_size, hidden_size))
        self.node_mlp_2 = Sequential(
            Linear(
                num_features +
                hidden_size +
                u_size,
                hidden_size),
            ReLU(),
            Linear(
                hidden_size,
                num_features))

    def forward(self, x, edge_index, u, batch, edge_attr=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = x[row]  # torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # torch.cat([x, out], dim=1) #torch.cat([x, out, u[batch]], dim=1)
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size, out_size, u_size):
        super(GlobalModel, self).__init__()
        self.global_mlp = Sequential(
            Linear(
                num_features +
                u_size,
                hidden_size),
            ReLU(),
            Linear(
                hidden_size,
                out_size))

    def forward(self, x, edge_index, u, batch, edge_attr=None):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class Encoder(torch.nn.Module):
    def __init__(self, num_features):
        super(Encoder, self).__init__()
        torch.manual_seed(12345)

        assert num_features >= 6

        self.activation = ReLU()
        self.pooling = AvgPool1d(kernel_size=7)

        # check if architecture requested is meaningful
        self.num_features = (num_features // 3) * 3
        channels = self.num_features // 3

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
            self.pooling
        )

    def forward(self, x):

        rows = x.shape[0]  # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1]  # length of a signal
        x = x.reshape(rows, 1, cols)  # reshape to 1 channel

        # 1. Extract features from time series
        x1 = self.hf_encoder(x).squeeze()
        x2 = self.mf_encoder(x).squeeze()
        x3 = self.lf_encoder(x).squeeze()

        x = torch.cat([x1, x2, x3], 1)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, num_features_node, num_features_global, hidden_size):
        super(Decoder, self).__init__()
        torch.manual_seed(12345)

        self.activation = ReLU()

        # 1D Convolutions to extract features from a signal
        self.classifier = Sequential(Linear(num_features_global, hidden_size),
                                     self.activation,
                                     Linear(hidden_size, hidden_size),
                                     self.activation,
                                     Linear(hidden_size, 3),
                                     Softmax(dim=-1)
                                     )
        self.decoder = Sequential(Linear(num_features_node, hidden_size),
                                  self.activation,
                                  Linear(hidden_size, hidden_size),
                                  self.activation,
                                  Linear(hidden_size, num_features_node)
                                  )

    def forward(self, x, u):

        x = self.decoder(x)
        u = self.classifier(u)

        return x, u


class Processor(torch.nn.Module):
    def __init__(self, hops, num_features, hidden_size_node,
                 hidden_size_glob, out_size, u_size):
        super(Processor, self).__init__()
        torch.manual_seed(12345)

        self.hops = hops

        self.processor = ModuleList([])
        self.processor.append(
            MetaLayer(
                None,
                NodeModel(
                    num_features,
                    hidden_size_node,
                    u_size),
                GlobalModel(
                    num_features,
                    hidden_size_glob,
                    out_size,
                    u_size)))
        for _ in range(hops):
            self.processor.append(
                MetaLayer(
                    None,
                    NodeModel(
                        num_features,
                        hidden_size_node,
                        out_size),
                    GlobalModel(
                        num_features,
                        hidden_size_glob,
                        out_size,
                        out_size)))

    def forward(self, x, edge_index, u, batch):

        for op in self.processor:
            x, _, u = op(x, edge_index, u, batch)

        return x, u


class Pipeline(torch.nn.Module):
    def __init__(self, hops, num_features, hidden_size_node,
                 hidden_size_glob, out_size, u_size, hidden_size):
        super(Pipeline, self).__init__()
        torch.manual_seed(12345)

        self.encoder = Encoder(num_features)
        self.processor = Processor(
            hops,
            num_features,
            hidden_size_node,
            hidden_size_glob,
            out_size,
            u_size)
        self.decoder = Decoder(num_features, out_size, hidden_size)

    def forward(self, x, edge_index, u, batch):

        x = self.encoder(x)
        x, u = self.processor(x, edge_index, u, batch)
        x, u = self.decoder(x, u)

        return x, u
