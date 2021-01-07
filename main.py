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
parser.add_argument('--hidden_channels', type=int, default=64,
                    help='Number of hidden channels of graph convolution layers')
parser.add_argument('--num_features', type=int, default=64,
                    help='Number of features to extract from a EEG signal')
args = parser.parse_args()

# Use GPU if available
if args.cuda:
    device = torch.device("cuda" if args.cuda else "cpu")
else:
    device = "cpu"

# Load dataset
if args.dataset == 'left':
    dataset = torch.load("EEG_data/train_data.pt")    
elif args.dataset == 'right':
    raise NotImplementedError
elif args.dataset == 'both':
    raise NotImplementedError

dataset = dataset[0:100]

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
    
# Send model to GPU 
model = model.double().to(device)
model = torch.nn.DataParallel(model)

# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

criterion = torch.nn.CrossEntropyLoss()

# Train the model 
train_acc = 0
best_acc = 0
epoch = 0

while train_acc < 1:
    loss = train(model, optimizer, criterion, train_loader, epoch)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), str(args.model) + '_' + str(args.exp_name) + '.pt')
    if epoch%5==0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    if args.epochs > 0 and epoch > args.epochs:
        break