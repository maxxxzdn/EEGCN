import torch
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data
import torch.nn.functional as F

from model import GCN
from baselines import EEGNet, ShallowNet
from utils import train, test, init_weights

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
parser.add_argument('--key', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 1.')
parser.add_argument('--lr', type=float, default=5e-5,
                    help='learning rate')
parser.add_argument('--use_cuda', type=int, default=1,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--wandb', type=int, default=1,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN', 'EEGNet', 'ShallowNet', 'pGCN', 'shGCN'])
#GCN
parser.add_argument('--d_latent', type=int, default=90,
                    help='Number of features to extract from a EEG signal')
parser.add_argument('--d_hidden', type=int, default=60,
                    help='Number of hidden channels of graph convolution layers')
parser.add_argument('--kernel_size', type=int, default=30)
parser.add_argument('--n_mp', type=int, default=1,
                    help='Hop distance in graph to collect information from, >=1')
parser.add_argument('--n_cnn', type=int, default=3,
                    help='Number of 1D convolutions to extract features from a signal, >=2')
parser.add_argument('--activation', type=str, default='tanh',
                    choices=['leaky_relu', 'relu', 'tanh'],
                    help='Activation function to use, [Leaky ReLU, ReLU, Tanh]')
parser.add_argument('--pooling', type=str, default='max',
                    choices=['max', 'avg'],
                    help='Pooling strategy to use, [Max, Average]')
parser.add_argument('--p_dropout', type=float, default=0.,
                    help='Dropout probability')
parser.add_argument('--normalization', type=str, default='minmax',
                    choices=['minmax', 's', 'z', 'f'],)
parser.add_argument('--norm_enc', type=int, default=1,
                    choices=[0,1],)
parser.add_argument('--norm_proc', type=str, default='graph',
                    choices=['none', 'batch', 'graph', 'layer'],)
parser.add_argument('--aggregate', type=str, default='mean',
                    choices=['none', 'eq', 'mean', 'max'],)
#EEGNet
parser.add_argument('--F1', type=int, default=2)
parser.add_argument('--D', type=int, default=32)
#ShallowConvNet
parser.add_argument('--n_filters_time', type=int, default=20)
parser.add_argument('--n_filters_spat', type=int, default=20)
parser.add_argument('--filter_time_length', type=int, default=30)
parser.add_argument('--pool_time_length', type=int, default=20)
parser.add_argument('--pool_time_stride', type=int, default=10)
args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project="eegcn")
    wandb.run.name = args.exp_name
    wandb.config.update(args, allow_val_change=True)
        
if args.use_cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_dataset = torch.load('train_dataset_id')
val_dataset = torch.load('val_dataset_id')
test_dataset = torch.load('test_dataset_id')

edge_index = np.loadtxt('edge_index3.txt')
edge_index = torch.tensor(edge_index).long()
for data in train_dataset:
    data.edge_index = edge_index
    if args.normalization == 'minmax':
        data.x = ((data.x + 100)/200)
    elif args.normalization == 'z':
        data.x = ((data.x - data.x.mean())/data.x.std())
    elif args.normalization == 's':
        data.x = ((data.x - data.x.mean(-1).unsqueeze(1))/data.x.std(-1).unsqueeze(1))
    elif args.normalization == 'f':
        data.x = F.normalize(data.x)
for data in val_dataset:
    data.edge_index = edge_index
    if args.normalization == 'minmax':
        data.x = ((data.x + 100)/200)
    elif args.normalization == 'z':
        data.x = ((data.x - data.x.mean())/data.x.std())
    elif args.normalization == 's':
        data.x = ((data.x - data.x.mean(-1).unsqueeze(1))/data.x.std(-1).unsqueeze(1))
    elif args.normalization == 'f':
        data.x = F.normalize(data.x)
for data in test_dataset:
    data.edge_index = edge_index
    if args.normalization == 'minmax':
        data.x = ((data.x + 100)/200)
    elif args.normalization == 'z':
        data.x = ((data.x - data.x.mean())/data.x.std())
    elif args.normalization == 's':
        data.x = ((data.x - data.x.mean(-1).unsqueeze(1))/data.x.std(-1).unsqueeze(1))
    elif args.normalization == 'f':
        data.x = F.normalize(data.x)
    
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

if args.model == 'GCN':
    model = GCN(args).to(device)
elif args.model == 'EEGNet':
    model = EEGNet(args).to(device)
elif args.model == 'ShallowNet':
    model = ShallowNet(args).to(device)
elif args.model == 'pGCN':
    model = pGCN(args).to(device)
elif args.model == 'shGCN':
    model = shGCN(args).to(device)   
    
model.apply(init_weights)
print(model)
if args.wandb:
    wandb.watch(model)
    wandb.log({'n parameters': sum(p.numel() for p in model.parameters())})
else:
    wandb = False
    
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
criterion = torch.nn.BCELoss()

# Train the model 
key = args.key
if key == 0:
    key = random.randint(0,200)
    
train(model, optimizer, train_loader, val_loader, test_loader, criterion, epochs = args.epochs, wandb = wandb, key = key)
