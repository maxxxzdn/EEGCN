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

parser = argparse.ArgumentParser(description='EEG Signal Backward Euler ')
parser.add_argument(
    '--path_data',
    type=str,
    default='../EEG_data/data/*/*/*/*')
parser.add_argument('--exp_name', type=str, default='euler',
                    help='Name of the experiment')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train. 0 leads to training until reaching train accuracy = 1.')
parser.add_argument('--experiments', type=int, default=2000,
                    help='Number of experiments to train on')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Size of batch')
parser.add_argument('--classifier', type=str, default='classifier.pt',
                    help='path to saved parameters of the pretrained classifier')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 5e-5, 5e-3 if using sgd)')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay value for an optimizer')
parser.add_argument('--cuda', type=int, default=1,
                    choices=[0, 1],
                    help='Enables CUDA training')
parser.add_argument('--horovod', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Horovod parallelism')
parser.add_argument('--wandb', type=int, default=0,
                    choices=[0, 1],
                    help='Enables Weights and Biases tracking')
parser.add_argument('--num_features', type=int, default=64,
                    help='Number of fetures assigned to each node')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='Dimension the graph latent space')
parser.add_argument('--hops', type=int, default=4,
                    help='Number of message passing steps, >=1')
parser.add_argument('--graph_info', type=str, default="../EEG_data/edge_index3.txt",
                    help='File with infomation about node labels and their positions')
args = parser.parse_args()

# Use GPU if available and requested
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    if args.horovod:  # Initialize horovod if requested
        import horovod.torch as hvd
        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
else:
    device = torch.device("cpu")

edge_index = np.loadtxt(args.graph_info)
edge_index = torch.tensor(edge_index).long()

model = Pipeline(args.hops, args.num_features, args.hidden_size)
model = model.double().to(device)

model.load_state_dict(torch.load('classifier.pt'))

# Load dataset
experiments = list(glob.glob(args.path_data))[:args.experiments]

train_dataset = TrajectoryDataset(experiments, edge_index)

# Split dataset into two: one for training the model and one for testing it
val_dataset = train_dataset[int(0.85 * (len(train_dataset))):]
train_dataset = train_dataset[:int(0.85 * (len(train_dataset)))]

# Initialize optimizer
optimizer = torch.optim.Adam(
    list(model.processor.parameters()) + list(model.decoder.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay)

if args.horovod:
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        backward_passes_per_step=1)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

# Train the model
epoch = 1

while train_acc < 0.99:
    loss = 0
    MSE_train = 0
    for i in range(0, len(train_dataset), args.batch_size):
        loss += train(model,
                      optimizer,
                      train_dataset[i:(i + args.batch_size)],
                      device)
        with torch.no_grad():
            MSE_train += test(model,
                              train_dataset[i:(i + args.batch_size)],
                              device)
    with torch.no_grad():
        MSE_val = test(model, val_dataset, device)
    if args.wandb:  # Write down train/val accuracies, MSE values and loss
        if not args.horovod or (args.horovod and hvd.rank()) == 0:
            wandb.log({"Train Accuracy": train_acc,
                       "Validation Accuracy": val_acc,
                       "Validation Loss": loss,
                       "Train MSE": MSE_train,
                       "Validation MSE": MSE_val,
                       "Epoch": epoch})
    else:
        if epoch % 1 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}, MSE Train: {MSE_train:.4f}, MSE Val: {MSE_val:.4f}')
    if not args.horovod or hvd.rank() == 0:
        if val_acc > best_acc:  # Save the best model and its accuracy result
            best_acc = val_acc
            if args.wandb:
                wandb.log({"Best model accuracy": best_acc})
    # If countable epoch number was given:
    if args.epochs > 0 and epoch > args.epochs:
        break

    epoch += 1
