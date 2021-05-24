import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Conv1d, MaxPool1d, AvgPool1d, ModuleList, Dropout, BatchNorm1d, CosineSimilarity, Parameter
from utils import corr2_coeff, init_weights
from my_conv import My_Conv
from mne.connectivity import spectral_connectivity

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        torch.manual_seed(12345)

        self.layers= ModuleList([])
        self.layers.append(Linear(61*200, 1000))
        self.layers.append(Linear(1000, 256))
        for _ in range(4):
            self.layers.append(Linear(256, 256))
        self.layers.append(Linear(256, 6))

    def forward(self, x, edge_index, batch):
        x = x.reshape((batch.max().item()+1),-1)     
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)    
        return F.log_softmax(x, -1) 

class CNN_MLP(torch.nn.Module):
    def __init__(self):
        super(CNN_MLP, self).__init__()
        torch.manual_seed(12345)
        self.activation = F.relu
        self.pooling = AvgPool1d(kernel_size = 7)
        self.convs1d = ModuleList([])
        self.convs1d.append(Conv1d(in_channels = 1, out_channels = 128, kernel_size = 7))
        for _ in range(2):
            self.convs1d.append(Conv1d(in_channels = 128, out_channels = 128, kernel_size = 5, padding = 2))
        self.convs1d.append(Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3))

        self.layers = ModuleList([])
        self.layers.append(Linear(61*128, 256))
        for _ in range(2):
            self.layers.append(Linear(256, 256))
        self.layers.append(Linear(256, 6))

    def forward(self, x, edge_index, batch):

        rows = x.shape[0] # number of signals in batch = 61 * graphs in batch
        cols = x.shape[1] # length of a signal
        x = x.reshape(rows,1,cols) # reshape to 1 channel

        x = self.convs1d[0](x)
        x = self.activation(x)
        x = self.pooling(x)
        x = F.dropout(x, p=0.25, training = self.training)
        
        for conv1d in self.convs1d[1:-1]:
            x = conv1d(x)
            x = self.activation(x)
            x = F.dropout(x, p=0.25, training = self.training)
            
        x = self.pooling(x)
        x = F.dropout(x, p=0.25, training = self.training)
        x = self.convs1d[-1](x)
        x = x.squeeze()

        x = x.reshape((batch.max().item()+1),-1)     
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)    
        return F.log_softmax(x, -1) 


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, n_classes, hops, layers, convs, pooling, aggr, corr, device):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        
        # check if architecture requested is meaningful
        assert hops > 0
        assert layers > 0
        assert convs > 1
        assert aggr in ['max', 'add']
        
        self.hops = hops
        self.layers = layers
        self.convs = convs
        self.n_classes = n_classes
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.activation = F.relu
        self.gconv = MyzConv #GraphConvBias
        self.aggr = aggr
        self.corr = corr
        
        #self.node_bias = Parameter(torch.tensor(61*[-10.], requires_grad=True, device = device).reshape(-1,1))
      
        if pooling == "max":
            self.pooling = MaxPool1d(kernel_size = 7)
            self.graphPooling = global_max_pool
        elif pooling == "avg":
            self.pooling = AvgPool1d(kernel_size = 7)
            self.graphPooling = global_mean_pool
        else:
            raise NotImplementedError          
                    
        # 1D Convolutions to extract features from a signal                    
        self.convs1d = ModuleList([])
        self.convs1d.append(Conv1d(in_channels = 1, out_channels = self.num_features, kernel_size = 7))
        for _ in range(self.convs-2):
            self.convs1d.append(Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 5, padding = 2))
        self.convs1d.append(Conv1d(in_channels = self.num_features, out_channels = self.num_features, kernel_size = 3)) 
        
        #self.convs1d = self.convs1d.apply(init_weights)
       
        self.bn1 = BatchNorm1d(self.num_features)
        
        #self.bn1 = self.bn1.apply(init_weights)
            
        # Graph Convolution Networks   
        self.gconvs = ModuleList([])
        self.gconvs.append(self.gconv(self.num_features, self.hidden_channels, aggr = self.aggr))
        for _ in range(self.hops-1):
            self.gconvs.append(self.gconv(self.hidden_channels, self.hidden_channels, aggr = self.aggr))
            
        self.bn2 = BatchNorm1d(self.hidden_channels)
        
        #self.bn2 = self.bn2.apply(init_weights)
        
        # Linear layers to make a prediction  
        self.linlayers = ModuleList([])
        self.linlayers.append(Linear(61*self.hidden_channels, self.hidden_channels))
        for _ in range(self.layers-2):
            self.linlayers.append(Linear(self.hidden_channels, self.hidden_channels))
        self.linlayers.append(Linear(self.hidden_channels, self.n_classes))    
        
        self.edge_net = torch.nn.Linear(3, 1)
        #torch.nn.init.ones_(self.edge_net.weight)
        #init_weights(self.edge_net)
        
        #self.linlayers = self.linlayers.apply(init_weights)

    def forward(self, x, edge_index, batch, edge_features):
        
        edge_weight = self.edge_net(edge_features) #.reshape(batch.max().item()+1,-1).T
        #edge_weight = (edge_weight - edge_weight.min(dim = 0)[0])/(edge_weight.max(dim = 0)[0] - edge_weight.min(dim = 0)[0])
        #edge_weight = edge_weight.T.reshape(-1)
        #edge_weight = F.relu(edge_weight)
        edge_weight = torch.tanh(edge_weight)
        #edge_weight = torch.nn.functional.softmax(edge_weight, 0).T.reshape(-1)
        
        """
        if self.corr: 
            indices = (edge_index[0].cpu().detach().numpy(), edge_index[1].cpu().detach().numpy())   # col indices
            data = [x.cpu().detach().numpy()]
            con_flat = spectral_connectivity(data, method='imcoh',
                                     indices=indices)
            edge_weight = np.abs(con_flat[0].mean(1))/np.abs(con_flat[0].mean(1)).max()
            edge_weight = torch.tensor(edge_weight).double().cuda()
        """    
            
        """
        edge_weight = None
        if self.corr:      
            x_j = x[edge_index[0]]
            x_i = x[edge_index[1]]
            edge_weight = 1 - corr2_coeff(x_j, x_i).abs()
        """

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
        x = x.squeeze()

        x = self.bn1(x)
        #x = F.dropout(x, p=0.33, training = self.training)
        #batch_max = (batch.max().item()+1)
        #x = self.activation(x)
        # 2. Obtain node embeddings; x.shape = [61,node_features]
        for gconv in self.gconvs[:-1]:
            x = gconv(x = x, edge_index = edge_index, edge_weight = edge_weight) #[61, hidden_channels]
            x = self.activation(x)
        x = self.gconvs[-1](x = x, edge_index = edge_index, edge_weight = edge_weight)
   
        x = self.bn2(x)
        #x = self.activation(x)
        # 3. Readout layer
        x = x.reshape((batch.max().item()+1),-1)
        
        # 4. Apply a final classifier
        #x = F.dropout(x, p=0.33, training=self.training)
        for linlayer in self.linlayers[:-1]:
            x = linlayer(x)
            x = self.activation(x)
        x = self.linlayers[-1](x)
        
        return F.log_softmax(x, -1), edge_weight.reshape(batch.max().item()+1,-1).T
