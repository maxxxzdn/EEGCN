import torch
import glob
import argparse
import numpy as np
import random
import torch.nn.functional as F
from model import *
from utils import *
from TrajectoryDataset import TrajectoryDataset

parser = argparse.ArgumentParser(description='EEG Signal Classification')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='Name of the experiment')
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
parser.add_argument('--wandb', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')

parser.add_argument('--num_features', type=int, default=9)
parser.add_argument('--hidden_size_node', type=int, default=10)
parser.add_argument('--hidden_size_glob', type=int, default=10)
parser.add_argument('--out_size_glob', type=int, default=10)
parser.add_argument('--out_size', type=int, default=5)
parser.add_argument('--hidden_size', type=int, default=10)

parser.add_argument('--hops', type=int, default=8,
                    help='Hop distance in graph to collect information from, >=1')

parser.add_argument('--graph_info', type=str, default="",
                    help='File with infomation about node labels and their positions')

args = parser.parse_args()

# Use GPU if available and requested
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load dataset
experiments = list(glob.glob('../EEG_data/Control/High/Control/Left/00015/*'))
edge_index = np.loadtxt(args.graph_info)
edge_index = torch.tensor(edge_index).long()

train_dataset = TrajectoryDataset(experiments, edge_index)

# Split dataset into two: one for training the model and one for testing it
val_dataset = train_dataset[int(0.85 * (len(train_dataset))):]
train_dataset = train_dataset[:int(0.85 * (len(train_dataset)))]

# Initialize the model
u_size = 2
model = Pipeline(
    args.hops,
    args.num_features,
    args.hidden_size_node,
    args.hidden_size_glob,
    args.out_size_glob,
    u_size,
    args.hidden_size)

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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

# Train the model
train_acc = 0
best_acc = 0
epoch = 1

while train_acc < 0.99:
    loss = train(model, optimizer, train_dataset, device)
    MSE_train, train_acc = test(model, train_dataset, device)
    MSE_val, test_acc = test(model, val_dataset, device)
    if args.wandb:  # Write down train/test accuracies and loss
        wandb.log({"Train Accuracy": train_acc,
                   "Test Accuracy": test_acc,
                   "Test Loss": loss,
                   "Epoch": epoch})
    else:
        if epoch % 1 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, MSE Train: {MSE_train:.4f}, MSE Val: {MSE_val:.4f}')
        if test_acc > best_acc:  # Save the best model and its accuracy result
            best_acc = test_acc
            if args.wandb:
                wandb.log({"Best model accuracy": best_acc})
    # If countable epoch number was given:
    if args.epochs > 0 and epoch > args.epochs:
        break

    epoch += 1

# Print best achieved result
if args.wandb:
    wandb.log({"Best model accuracy": best_acc})
else:
    print("Best model accuracy: " + str(round(best_acc, 2)))
