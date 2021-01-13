import torch
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, hops = 4, layers = 2, convs = 3, activation = 'relu', pooling = "avg"):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        
        assert hops > 0
        assert layers > 0
        assert convs > 1
        
        self.hops = hops
        self.layers = layers
        self.convs = convs

        self.num_features = num_features
        self.activation = activation
        
        if activation == 'leaky_relu':
        	self.activation = F.leaky_relu
        elif activation == 'relu':
                self.activation = F.relu
        elif activation == 'tanh':
                self.activation = F.tanh
        else: 
            raise NotImplementedError
        
        if pooling == "max":
            self.pooling = MaxPool1d(kernel_size = 5)
            self.graphPooling = global_max_pool
        elif pooling == "avg":
            self.pooling = AvgPool1d(kernel_size = 5)
            self.graphPooling = global_mean_pool
        else:
            raise NotImplementedError
                    
        # 1D Convolutions to extract features from a signal            
        self.convs1d = ModuleList([])
        self.convs1d.append(Conv1d(in_channels = 1, out_channels = self.num_features, kernel_size = 7))
        for _ in range(self.convs-2):
            self.convs1d.append(Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 5, padding = 2))
        self.convs1d.append(Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 3))
        
        # Graph Convolution Networks
        self.gconvs = ModuleList([])
        self.gconvs.append(GraphConv(self.num_features, hidden_channels))
        for _ in range(self.hops-1):
            self.gconvs.append(GraphConv(hidden_channels, hidden_channels))
	
	# Linear layers to make a prediction  
        self.linlayers = ModuleList([])
        for _ in range(self.layers-1):
            self.linlayers.append(Linear(hidden_channels, hidden_channels))
        self.linlayers.append(Linear(hidden_channels, 2))


    def forward(self, x, edge_index, batch):

        rows = x.shape[0] # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1] # length of a signal
        x = x.reshape(rows,1,cols) # reshape to 1 channel

        # 1. Extract features from time series
        x = self.convs1d[0](x)
        x = self.activation(x)
        x = self.pooling(x)
        
        for conv1d in self.convs1d[1:-1]:
            x = conv1d(x)
            x = self.activation(x)
        x = self.pooling(x)
        
        x = self.convs1d[-1](x)
        x = self.activation(x)
        x = self.pooling(x)

        x = x.squeeze()

        # 2. Obtain node embeddings; x.shape = [61,node_features]
        for gconv in self.gconvs:
            x = gconv(x, edge_index) #[61, hidden_channels]
            x = self.activation(x)
   
        # 3. Readout layer
        x = self.graphPooling(x, batch)  # [batch_size, hidden_channels]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        for linlayer in self.linlayers[:-1]:
            x = linlayer(x)
            x = self.activation(x)
        x = self.linlayers[-1](x)
        
        return x
