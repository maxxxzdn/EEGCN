import torch
import glob
import argparse
import numpy as np
import random
import torch.nn.functional as F
from model import *
from utils import *
from TrajectoryDataset import TrajectoryDataset
from torch_geometric.data import DataLoader

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='classifier',
                    help='Name of the experiment')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 0.99.')
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--train_dataset', type=str, default='../EEG_data/data/train_data.pt',
                    help='Data to train on')
parser.add_argument('--val_dataset', type=str, default='../EEG_data/data/val_data.pt',
                    help='Data to validate on')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay value for an optimizer')
parser.add_argument('--cuda', type=int, default=1,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--horovod', type=int, default=1,
                    choices=[0, 1],
                    help='Enables Horovod parallelism')
parser.add_argument('--wandb', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')
parser.add_argument('--num_features', type=int, default=64, help='Number of fetures assigned to each node')
parser.add_argument('--hidden_size', type=int, default=128, help='Dimension the graph latent space')
parser.add_argument('--hops', type=int, default=4,
                    help='Number of message passing steps, >=1')
parser.add_argument('--graph_info', type=str, default="../EEG_data/edge_index3.txt",
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
    
train_dataset = torch.load(args.train_dataset)
val_dataset = torch.load(args.val_dataset)
random.shuffle(train_dataset)
random.shuffle(val_dataset)

edge_index = np.loadtxt(args.graph_info)
edge_index = torch.tensor(edge_index).long()

for data in train_dataset:
    data.edge_index = edge_index
for data in val_dataset:
    data.edge_index = edge_index
    
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

model = Pipeline(args.hops, args.num_features, args.hidden_size)
model = model.double().to(device)    

if args.wandb:
    if not args.horovod or (args.horovod and hvd.rank()) == 0:
        import wandb
        wandb.init(project="eegcn")
        wandb.run.name = args.exp_name
        wandb.config.update(args, allow_val_change=True)
        wandb.watch(model)
    
optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

if args.horovod:
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.classifier.named_parameters(), backward_passes_per_step=1)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

criterion = torch.nn.CrossEntropyLoss()

epoch = 0
best_acc = 0

while epoch < args.epochs:
    loss = cl_train(model.classifier, optimizer, criterion, train_loader, epoch, device)
    with torch.no_grad():
        train_acc = cl_test(model.classifier, train_loader, device)
        test_acc = cl_test(model.classifier, val_loader, device)    
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
    epoch += 1
