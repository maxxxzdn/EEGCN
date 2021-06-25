import torch
#torch.manual_seed(12345)
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
parser.add_argument('--graph_info', type=str, default="./datasets/Easycap_Koordinaten_61CH.txt",
                    help='File with infomation about node labels and their positions')
#Learning parameters
parser.add_argument('--batch_size', type=int, default=32, 
                    help='Size of batch')
parser.add_argument('--epochs', type=int, default=100,
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
parser.add_argument('--num_features', type=int, default=32,
                    help='Number of features to extract from a EEG signal')
parser.add_argument('--hops', type=int, default=1,
                    help='Hop distance in graph to collect information from, >=1')          
parser.add_argument('--aggr', type=str, default='max',
                    choices=['max', 'add', 'mean'],
                    help='The aggregation scheme to use in graph convolution')
parser.add_argument('--lambda_', type=float, default=1.,
                    help='Lambda for adjacency loss')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Max distance between 2 nodes to account for the edge')
parser.add_argument('--n', type=int, default=4,
                    help='Number of layers in Encoder/Decoder')
parser.add_argument('--kernel_size', type=int, default=11,
                    help='Kernel size of convolution opearators in ConvAE')
parser.add_argument('--autoencoder', type=str, default='ffn',
                    choices=['ffn', 'conv'],
                    help='Autoencoder type')
args = parser.parse_args()

# Use GPU if available and requested
if args.cuda and torch.cuda.is_available():
    device = torch.device("cuda")
    #torch.cuda.manual_seed_all(12345)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
# Load dataset
train_dataset = torch.load(args.train_dataset)
random.shuffle(train_dataset)
val_dataset = torch.load(args.val_dataset)
random.shuffle(val_dataset)
train_dataset = [data for data in train_dataset if data.y.item() in [0,1]]
val_dataset = [data for data in val_dataset if data.y.item() in [0,1]]

for data in train_dataset:
    data.y = data.y%2
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

# Add connectivity information and edge features to data 
#M = torch.cat([data.x for data in train_dataset],0).reshape(-1, 61, 200).mean()
#S = torch.cat([data.x for data in train_dataset],0).reshape(-1, 61, 200).std()
   
# Create batches with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker)

# Initialize the model 
model = make_model(args.hops, args.num_features, args.aggr, positions, args.threshold, args.autoencoder, args.kernel_size)
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
test_loss = 1.
epoch = 0
criterion = torch.nn.CrossEntropyLoss()

while epoch<-1:
    _ = train_ae(model, optimizer, train_loader, device)
    train_acc = test_ae(model, train_loader, device)
    test_acc= test_ae(model, val_loader, device)
    if epoch%5==0:
        print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Train Acc: {train_acc:.4f}')  
    epoch += 1
        
epoch = 1

while train_acc < 0.99:
    _ = train(model, optimizer, criterion, train_loader, epoch, device, args.lambda_)
    train_acc, train_loss, train_ras = test(model, train_loader, device, criterion)
    test_acc, test_loss, test_ras = test(model, val_loader, device, criterion)
    if test_ras > best_ras: # Save the best model and its accuracy result
        best_loss = test_loss
        best_acc = test_acc
        best_ras = test_ras
        if args.wandb:
            wandb.log({"Best model loss": best_loss, "Best model accuracy": best_acc, "Best model accuracy": best_ras, "Epoch": epoch})
        torch.save(model.state_dict(), './models/' + 'gcn' + '_' + str(args.exp_name) + '.pt')
    if args.wandb: # Write down train/test accuracies and loss
        wandb.log({"Train Accuracy": train_acc, "Test Accuracy": test_acc, "Train Loss": train_loss, "Test Loss": test_loss, "Train RAS": train_ras, "Test RAS": test_ras, "Epoch": epoch})
    else: 
        if epoch%500==0:
            print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Diff: {test_acc/train_acc:.2f}, Best Acc: {best_acc:.4f}')           
    if args.epochs > 0 and epoch > args.epochs: # If countable epoch number was given:
        break        
    epoch += 1
# Print best achieved result
if args.wandb:
    wandb.log({"Best model RAS": best_ras, "Epoch": epoch})
else:
    print("Best model RAS: " + str(round(best_ras,4)))
