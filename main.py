import torch
import glob
import argparse

import numpy as np

from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from torch.nn import Linear, Conv1d, MaxPool1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool, global_sort_pool, global_max_pool

from model import GCN
from utils import train, test

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
parser.add_argument('--data', type=str, default='EEG_data/train_data.pt',
                    help='Data to train on')
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'mp'],
                    help='Model to use, [GCN, MP]')
parser.add_argument('--dataset', type=str, default='left', 
                    choices=['right', 'left', 'both'])
parser.add_argument('--batch_size', type=int, default=64, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 1.')
parser.add_argument('--optimizer', type=str, default='adam', 
                    choices=['adam', 'sgd'],
                    help='Optimizer to use, [Adam, SGD]')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', type=bool, default=False,
                    help='Enables CUDA training')
parser.add_argument('--horovod', type=bool, default=False,
                    help='Enables Horovod parallelism')
parser.add_argument('--wandb', type=bool, default=False,
                    help='Enables Weights and Biases tracking')
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Number of hidden channels of graph convolution layers')
parser.add_argument('--num_features', type=int, default=64,
                    help='Number of features to extract from a EEG signal')
               
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
if args.dataset == 'left':
    dataset = torch.load(args.data)    
elif args.dataset == 'right':
    raise NotImplementedError
elif args.dataset == 'both':
    raise NotImplementedError

# Split dataset into two: one for training the model and one for testing it
train_dataset = dataset[:int(0.85*(len(dataset)))] 
test_dataset = dataset[int(0.85*(len(dataset))):]

# Create batches with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Initialize the model 
if args.model == 'gcn':
    model = GCN(hidden_channels=args.hidden_channels, num_features = args.num_features)
elif args.model == 'mp':
    raise NotImplementedError

# Start wandb tracking if requested
if args.wandb:
    import wandb
    if args.horovod:
        if hvd.rank() == 0:  
            wandb.init(project="eegcn")
            wandb.run.name = args.exp_name
            wandb.config.update(args, allow_val_change=True)
            wandb.watch(model)
    else: 
        wandb.init(project="eegcn")
        wandb.run.name = args.exp_name
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
   
# Send model to GPU or CPU
model = model.double().to(device)

# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

# Distribute parameters across resources
if args.horovod:
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), backward_passes_per_step=1)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

criterion = torch.nn.CrossEntropyLoss()

# Train the model 
train_acc = 0
best_acc = 0
epoch = 0

while train_acc < 1:
    loss = train(model, optimizer, criterion, train_loader, epoch, device)
    train_acc = test(model, train_loader, device)
    test_acc = test(model, test_loader, device)
    
    if args.wandb: 
        if args.horovod:
            if hvd.rank() == 0:
                wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Test Loss": loss, "Epoch": epoch})
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(model.state_dict(), str(args.model) + '_' + str(args.exp_name) + '.pt')
        else:
            wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Test Loss": loss, "Epoch": epoch})
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), str(args.model) + '_' + str(args.exp_name) + '.pt')
    else:
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), str(args.model) + '_' + str(args.exp_name) + '.pt')
        if epoch%5==0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


    if args.epochs > 0 and epoch > args.epochs:
        break

if args.wandb:
    if args.horovod:
        if hvd.rank() == 0:
            wandb.log({"Best model accuracy": best_acc})
    else: 
        wandb.log({"Best model accuracy": best_acc})
else:
    print("Best model accuracy: " + str(round(best_acc,2)))
