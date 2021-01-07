import torch
from torch.nn import Linear, Conv1d, MaxPool1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool, global_sort_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.num_features = num_features
        self.c1 = Conv1d(in_channels = 1, out_channels = self.num_features, kernel_size = 5)
        self.mp1 = MaxPool1d(kernel_size = 5)
        self.c2 = Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 5)
        self.mp2 = MaxPool1d(kernel_size = 5)
        self.c3 = Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 5)
        self.mp3 = MaxPool1d(kernel_size = 3)
        
        self.conv1 = GraphConv(self.num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch):
        
        rows = x.shape[0] # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1] # length of a signal
        x = x.reshape(rows,1,cols) # reshape to 1 channel
        
        # 1. Extract features from time series
        x = self.c1(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = self.mp2(x)
        x = self.c3(x)
        x = self.mp3(x)
    
        x = x.squeeze()    
        # 2. Obtain node embeddings 
        # x.shape = [61,node_features]
        x = self.conv1(x, edge_index) #[61, hidden_channels]
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index) #[61, hidden_channels]
        x = F.leaky_relu(x)
        x = self.conv3(x, edge_index) #[61, hidden_channels]
        x = F.leaky_relu(x)
        x = self.conv4(x, edge_index) #[61, hidden_channels]

        # 3. Readout layer
        x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin1(x)
        x = F.leaky_relu(x)
        x = self.lin2(x)
        
        return x