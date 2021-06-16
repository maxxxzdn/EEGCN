import torch
torch.manual_seed(12345)
import glob
import argparse
import random
random.seed(12345)
import numpy as np
np.random.seed(12345)
from torch_geometric.data import DataLoader
from model import *
from utils import *
from mne.connectivity import spectral_connectivity

parser = argparse.ArgumentParser(description='EEG Signal Classification')
#Data parameters
parser.add_argument('--train_dataset', type=str, default='datasets/train_data_6.pt',
                    help='Data to train on')
parser.add_argument('--val_dataset', type=str, default='datasets/val_data_6.pt',
                    help='Data to validate on')
parser.add_argument('--edge_index', type=str, default='datasets/edge_index3.txt',
                    help='Conenctivity data')
parser.add_argument('--graph_info', type=str, default="./datasets/Easycap_Koordinaten_61CH.txt",
                    help='File with infomation about node labels and their positions')
#Learning parameters
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'cnn', 'mlp', 'gcn_st', 'gcn_cst'],
                    help='Model to use, [GCN, CNN+MLP, MLP]')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train')
parser.add_argument('--optimizer', type=str, default='adamW', 
                    choices=['adam', 'wadam', 'sgd'],
                    help='Optimizer to use, [Adam, AdamW, SGD]')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', type=int, default=1,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--wandb', type=int, default=1,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
#Model parameters
parser.add_argument('--hidden_channels', type=int, default=128,
                    help='Number of hidden channels of graph convolution layers')
parser.add_argument('--num_features', type=int, default=64,
                    help='Number of features to extract from a EEG signal')
parser.add_argument('--pooling', type=str, default='avg',
                    choices=['max', 'avg'],
                    help='Pooling strategy to use, [Max, Average]')
parser.add_argument('--activation', type=str, default='relu',
                    choices=['leaky_relu', 'relu', 'tanh'],
                    help='Activation function to use, [Leaky ReLU, ReLU, Tanh]')
parser.add_argument('--hops', type=int, default=1,
                    help='Hop distance in graph to collect information from, >=1')
parser.add_argument('--layers', type=int, default=2,
                    help='Classification layers in the model, >=1')                  
parser.add_argument('--convs', type=int, default=4,
                    help='Number of 1D convolutions to extract features from a signal, >=2')                
parser.add_argument('--aggr', type=str, default='max',
                    choices=['max', 'add'],
                    help='The aggregation scheme to use in graph convolution')
parser.add_argument('--edge_strategy', type=str, default='none',
                    choices=['none', 'corr', 'edge_net'],
                    help='Strategy to get edge weights')
parser.add_argument('--graph_operator', type=str, default='my_conv',
                    choices=['gat', 'my_conv', 'graph_conv', 'cheb'],
                    help='Graph operator to use')
parser.add_argument('--dropout', type=float, default=0.33,
                    help='Probability of an element to be zeroed')
parser.add_argument('--lambda_', type=float, default=1.,
                    help='Lambda for adjacency loss')
parser.add_argument('--threshold', type=float, default=1.,
                    help='Max distance between 2 nodes to account for the edge')
parser.add_argument('--n', type=int,
                    help='Model number')
args = parser.parse_args()

# Use GPU if available and requested
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(12345)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
# Load dataset
train_dataset = torch.load(args.train_dataset)
random.shuffle(train_dataset)
val_dataset = torch.load(args.val_dataset)
random.shuffle(val_dataset)
train_dataset = [data for data in train_dataset if data.y.item() in [0,1]]
for data in train_dataset:
    data.y = data.y%2
val_dataset = [data for data in val_dataset if data.y.item() in [0,1]]
for data in val_dataset:
    data.y = data.y%2
# Investigate number of classes in dataset and length of signal
classes = []
for data in train_dataset[:100]:
    if data.y.item() not in classes:
        classes.append(data.y.item())
n_classes = len(classes)
signal_length = data.x.shape[1]
# Extract connectivity information
positions = np.genfromtxt(args.graph_info)[:,0:3]
positions = torch.tensor(positions).to(device)
if args.edge_index == 'none':
    edge_index = [[],[]]
    for i in range(61):
        for j in range(61):
            edge_index[0].append(i)
            edge_index[1].append(j)
else:
    edge_index = np.loadtxt(args.edge_index)
edge_index = torch.tensor(edge_index).long()
# Add connectivity information and edge features to data 
for data in train_dataset:
    data.edge_index = edge_index
    if args.edge_strategy == 'edge_net':    
        indices = (edge_index[0], edge_index[1])   # col indices
        x = [data.x.cpu().detach().numpy()]
        con_flat = spectral_connectivity(x, method='imcoh', indices=indices, verbose = False)
        con_flat = np.abs(con_flat[0]).mean(1)
        con_flat = torch.tensor(con_flat).double().reshape(1,-1)
        dist = torch.linalg.norm(positions[edge_index[0]]-positions[edge_index[1]],2, 1).reshape(1,-1)
        de = (1/2*torch.log(2*np.pi*np.exp(1)*data.x.var(1)**2))
        dasm = (torch.abs(de[edge_index[0]] - de[edge_index[1]])).reshape(1,-1).double()
        data.edge_features = torch.cat((dist, con_flat, dasm),0).T
for data in val_dataset:
    data.edge_index = edge_index
    if args.edge_strategy == 'edge_net':     
        indices = (edge_index[0], edge_index[1])   # col indices
        x = [data.x.cpu().detach().numpy()]
        con_flat = spectral_connectivity(x, method='imcoh', indices=indices, verbose = False)
        con_flat = np.abs(con_flat[0]).mean(1)
        con_flat = torch.tensor(con_flat).double().reshape(1,-1)
        dist = torch.linalg.norm(positions[edge_index[0]]-positions[edge_index[1]],2, 1).reshape(1,-1)
        de = (1/2*torch.log(2*np.pi*np.exp(1)*data.x.var(1)**2))
        dasm = (torch.abs(de[edge_index[0]] - de[edge_index[1]])).reshape(1,-1).double()
        data.edge_features = torch.cat((dist, con_flat, dasm),0).T
# Create batches with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)
# Initialize the model 
"""
if args.model == 'gcn':
    model = GCN(hidden_channels=args.hidden_channels, num_features = args.num_features, n_classes = n_classes, hops = args.hops, layers = args.layers, convs = args.convs, pooling = args.pooling, aggr = args.aggr, edge_strategy = args.edge_strategy, graph_operator = args.graph_operator, dropout = args.dropout, attention = 0)
elif args.model == 'gcn_st':
    model = GCN_SnT(hidden_channels=args.hidden_channels, num_features = args.num_features, n_classes = n_classes, hops = args.hops, layers = args.layers, convs = args.convs, pooling = args.pooling, aggr = args.aggr, edge_strategy = args.edge_strategy, graph_operator = args.graph_operator, dropout = args.dropout, positions = positions.cuda().double())
elif args.model == 'gcn_cst':
    model = GCNСST(hidden_channels=args.hidden_channels, num_features = args.num_features, n_classes = n_classes, hops = args.hops, layers = args.layers, convs = args.convs, pooling = args.pooling, aggr = args.aggr, edge_strategy = args.edge_strategy, graph_operator = args.graph_operator, dropout = args.dropout)
elif args.model == 'mlp':
    model = MLP()
elif args.model == 'cnn':
    model = CNN_MLP()
"""
model = make_model(args.convs, args.hops, args.num_features, args.hidden_channels, args.aggr, args.pooling, positions, args.threshold, args.dropout, n=args.n)
# Start wandb tracking if requested
if args.wandb:
    import wandb
    wandb.init(project="eegcn")
    wandb.run.name = args.exp_name
    wandb.config.update(args, allow_val_change=True)
    wandb.config.update({"n_classes": n_classes})
    wandb.watch(model)
# Send model to GPU or CPU
model = model.double().to(device)
# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'adamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,  weight_decay=args.lr)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
# Train the model 
best_ras = 0.5
best_acc = 0.0
train_acc = 0
epoch = 1
criterion = torch.nn.CrossEntropyLoss()

while train_acc < 0.999:
    _ = train(model, optimizer, criterion, train_loader, epoch, device, args.lambda_)
    train_acc, train_loss, train_ras = test(model, train_loader, device, criterion)
    test_acc, test_loss, test_ras = test(model, val_loader, device, criterion)
    if test_ras > best_ras: # Save the best model and its accuracy result
        best_loss = test_loss
        best_acc = test_acc
        best_ras = test_ras
        if args.wandb:
            wandb.log({"Best model loss": best_loss, "Best model accuracy": best_acc, "Best model accuracy": best_ras, "Epoch": epoch})
        torch.save(model.state_dict(), './models/' + str(args.model) + '_' + str(args.exp_name) + '.pt')
    if args.wandb: # Write down train/test accuracies and loss
        wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Train Loss": train_loss, "Test Loss": test_loss, "Train RAS": train_ras, "Test RAS": test_ras, "Epoch": epoch})
    else: 
        if epoch%5==0:
            print(f'Epoch: {epoch:03d}, Test RAS: {test_ras:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Best Acc: {best_acc:.4f},  Best RAS: {best_ras:.4f}')           
    if args.epochs > 0 and epoch > args.epochs: # If countable epoch number was given:
        break        
    epoch += 1
# Print best achieved result
if args.wandb:
    wandb.log({"Best model RAS": best_ras, "Epoch": epoch})
else:
    print("Best model RAS: " + str(round(best_ras,4)))
