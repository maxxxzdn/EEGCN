import torch
import glob
import argparse
import random
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
from model import *
from utils import train, test

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
parser.add_argument('--train_dataset', type=str, default='datasets/train_data_6.pt',
                    help='Data to train on')
parser.add_argument('--val_dataset', type=str, default='datasets/val_data_6.pt',
                    help='Data to validate on')
parser.add_argument('--edge_index', type=str, default='edge_index3.txt',
                    help='Conenctivity data')
parser.add_argument('--model', type=str, default='gcn',
                    choices=['gcn', 'cnn', 'mlp'],
                    help='Model to use, [GCN, CNN+MLP, MLP]')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 1.')
parser.add_argument('--optimizer', type=str, default='adam', 
                    choices=['adam', 'wadam', 'sgd'],
                    help='Optimizer to use, [Adam, SGD]')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 5e-5, 5e-3 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', type=int, default=1,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--horovod', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Horovod parallelism')
parser.add_argument('--wandb', type=int, default=1,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')

parser.add_argument('--graph_pool', type=int, default=0,
                    help='Pool the graph or not')
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
parser.add_argument('--layers', type=int, default=4,
                    help='Classification layers in the model, >=1')                  
parser.add_argument('--convs', type=int, default=6,
                    help='Number of 1D convolutions to extract features from a signal, >=2')                
parser.add_argument('--graph_info', type=str, default="Easycap_Koordinaten_61CH.txt",
                    help='File with infomation about node labels and their positions')
parser.add_argument('--aggr', type=str, default='add',
                    choices=['max', 'add'])
parser.add_argument('--corr', type=int, default=0,
                    choices=[0,1])
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
    
val_dataset = torch.load(args.val_dataset)
random.shuffle(val_dataset)

#train_dataset = [data for data in train_dataset if data.y.item() in [0,1]]
#val_dataset = [data for data in val_dataset if data.y.item() in [0,1]]
# Investigate number of classes in dataset and length of signal
classes = []
for data in train_dataset[:100]:
    if data.y.item() not in classes:
        classes.append(data.y.item())
n_classes = len(classes)
signal_length = data.x.shape[1]

# Add graph information to data
positions = np.genfromtxt(args.graph_info)[:,0:3]
positions = torch.tensor(positions)

edge_index = np.loadtxt(args.edge_index)
edge_index = torch.tensor(edge_index).long()

for data in train_dataset:
    data.edge_index = edge_index
    data.positions = positions
for data in val_dataset:
    data.edge_index = edge_index
    data.positions = positions

# Create batches with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

# Initialize the model 
if args.model == 'gcn':
    model = GCN(hidden_channels=args.hidden_channels, num_features = args.num_features, n_classes = n_classes, hops = args.hops, layers = args.layers, convs = args.convs, pooling = args.pooling, aggr = args.aggr, corr = args.corr, device = device)
elif args.model == 'mlp':
    model = MLP()
elif args.model == 'cnn':
    model = CNN_MLP()

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

my_list = ['node_bias']
params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))

# Initialize optimizer
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam([{'params': base_params, 'lr': args.lr}, {'params': params, 'lr': args.lr}])
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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

while train_acc < 0.9999:
    loss = train(model, optimizer, criterion, train_loader, epoch, device)
    train_acc = test(model, train_loader, device)
    test_acc = test(model, val_loader, device) 
    if epoch%5==0:
        for label in range(n_classes):
            my_dataset = [data for data in val_dataset if data.y.item() == (label)]
            my_loader = DataLoader(my_dataset, batch_size=args.batch_size, shuffle=True)
            label_acc = test(model, my_loader, device)

            if args.wandb: # Write down train/test accuracies and loss
                if not args.horovod or (args.horovod and hvd.rank()) == 0:
                    wandb.log({str(label) + " Accuracy": label_acc, "Epoch": epoch})

   
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
        wandb.log({"Best model accuracy": best_acc, "Epoch": epoch})
else:
    print("Best model accuracy: " + str(round(best_acc,2)))
