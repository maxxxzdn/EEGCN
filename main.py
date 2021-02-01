import torch
import glob
import argparse
import numpy as np
import random
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch.nn import Linear, Conv1d, MaxPool1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv, GATConv
from torch_geometric.nn import global_mean_pool, global_sort_pool, global_max_pool
from model import GCN
from utils import train, test, distance, edgeCreator

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
parser.add_argument('--train_dataset', type=str, default='EEG_data/train_data.pt',
                    help='Data to train on')
parser.add_argument('--val_dataset', type=str, default='',
                    help='Data to validate on')
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'mp'],
                    help='Model to use, [GCN, MP]')
parser.add_argument('--graph_operator', type=str, default='GraphConv',
                    choices=['GCNConv', 'GraphConv', 'GATConv'],
                    help='Model to use, [GCN, MP]')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 1.')
parser.add_argument('--optimizer', type=str, default='adam', 
                    choices=['adam', 'sgd'],
                    help='Optimizer to use, [Adam, SGD]')
parser.add_argument('--lr', type=float, default=0.00005,
                    help='learning rate (default: 5e-5, 5e-3 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay value for an optimizer')
parser.add_argument('--cuda', type=int, default=0,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--horovod', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Horovod parallelism')
parser.add_argument('--wandb', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Number of hidden channels of graph convolution layers')
parser.add_argument('--num_features', type=int, default=64,
                    help='Number of features to extract from a EEG signal')
parser.add_argument('--pooling', type=str, default='avg',
                    choices=['max', 'avg'],
                    help='Pooling strategy to use, [Max, Average]')
parser.add_argument('--activation', type=str, default='leaky_relu',
                    choices=['leaky_relu', 'relu', 'tanh'],
                    help='Activation function to use, [Leaky ReLU, ReLU, Tanh]')
parser.add_argument('--hops', type=int, default=8,
                    help='Hop distance in graph to collect information from, >=1')
parser.add_argument('--layers', type=int, default=4,
                    help='Classification layers in the model, >=1')                  
parser.add_argument('--convs', type=int, default=5,
                    help='Number of 1D convolutions to extract features from a signal, >=2')
parser.add_argument('--explain', type=int, default=0,
                    choices=[0,1],
                    help='Calculate node importance using Integrated Gradients (captum)')                     
parser.add_argument('--graph_info', type=str, default="",
                    help='File with infomation about node labels and their positions')
args = parser.parse_args()

# Use GPU if available and requested
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    if args.horovod: # Initialize horovod if requested
        import horovod.torch as hvd
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
else:
    device = torch.device("cpu")

# Load dataset
train_dataset = torch.load(args.train_dataset)
random.shuffle(train_dataset)
    
# Split dataset into two: one for training the model and one for testing it
if len(args.val_dataset) == 0:
    train_dataset = train_dataset[:int(0.85*(len(train_dataset)))] 
    val_dataset = train_dataset[int(0.85*(len(train_dataset))):]
else: 
    val_dataset = torch.load(args.val_dataset)
    random.shuffle(val_dataset)

# Investigate number of classes in dataset and length of signal
classes = []
for data in train_dataset[:100]:
    if data.y.item() not in classes:
        classes.append(data.y.item())
n_classes = len(classes)
signal_length = data.x.shape[1]

# Add graph information to data
positions = np.genfromtxt(args.graph_info)[:,0:3]
labels = np.genfromtxt(args.graph_info, 'str')[:,3]
positions = torch.tensor(positions)

edge_index = np.loadtxt('./EEG_data/edge_index3.txt')
edge_index = torch.tensor(edge_index).long()

for data in train_dataset:
    data.edge_index = edge_index
for data in val_dataset:
    data.edge_index = edge_index
    
# Create batches with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model 
if args.model == 'gcn':
    model = GCN(hidden_channels=args.hidden_channels, num_features = args.num_features, n_classes = n_classes, graph_operator = args.graph_operator, hops = args.hops, layers = args.layers, convs = args.convs, activation = args.activation, pooling = args.pooling)
elif args.model == 'mp':
    raise NotImplementedError

# Start wandb tracking if requested
if args.wandb:
    import wandb
    if not args.horovod or (args.horovod and hvd.rank()) == 0:
        wandb.init(project="eegcn")
        wandb.run.name = args.exp_name
        wandb.config.update(args, allow_val_change=True)
        wandb.config.update({"n_classes": n_classes})
        wandb.watch(model)
   
# Send model to GPU or CPU
model = model.double().to(device)

# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Distribute parameters across resources
if args.horovod:
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), backward_passes_per_step=1)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

criterion = torch.nn.CrossEntropyLoss()

# Train the model 
train_acc = 0
best_acc = 0
epoch = 1

while train_acc < 0.99:
    loss = train(model, optimizer, criterion, train_loader, epoch, device)
    train_acc = test(model, train_loader, device)
    test_acc = test(model, val_loader, device)    
    if args.wandb: # Write down train/test accuracies and loss
        if not args.horovod or (args.horovod and hvd.rank()) == 0:
            wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Test Loss": loss, "Epoch": epoch})
    else: 
        if epoch%5==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')           
    if not args.horovod or hvd.rank() == 0:
        if test_acc > best_acc: # Save the best model and its accuracy result
            best_acc = test_acc
            if args.wandb:
                wandb.log({"Best model accuracy": best_acc})
            torch.save(model.state_dict(), str(args.model) + '_' + str(args.exp_name) + '.pt')
    if args.epochs > 0 and epoch > args.epochs: # If countable epoch number was given:
        break 
        
    epoch += 1

# Print best achieved result
if args.wandb:
    if not args.horovod or (args.horovod and hvd.rank()) == 0:
        wandb.log({"Best model accuracy": best_acc})
else:
    print("Best model accuracy: " + str(round(best_acc,2)))
    
# Explain graph network predictions using integrated gradients if requested
if args.explain:
    from utils import *
    positions = positions[:,:2]
    explain(model, dataset, positions, labels)
