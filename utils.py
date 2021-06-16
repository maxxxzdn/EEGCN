import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def train(model, optimizer, criterion, train_loader, epoch, device, lambda_ = 1.):
    model.train() # set training mode for the model             
    loss_all = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to('cuda')
        out, adj = model(x = data.x, edge_index = data.edge_index, batch_size = data.batch.max().item() + 1)  # Perform a single forward pass.     
        adj_loss = torch.nn.functional.l1_loss(adj, torch.zeros_like(adj))
        loss = criterion(out, data.y) + lambda_*adj_loss # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss_all += loss.item()       
    return loss_all/len(train_loader.dataset)

def test(model, loader, device, criterion):
    model.eval() # set evaluation mode for the model
    correct = 0
    loss_all = 0
    y = torch.tensor([])
    y_pred = torch.tensor([])
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out,_ = model(x = data.x, edge_index = data.edge_index, batch_size = data.batch.max().item() + 1)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        loss = criterion(out, data.y) #+ 0.001*loss_am # Compute the loss.
        loss_all += loss.item()     
        y = torch.cat([y, data.y.reshape(-1).cpu().detach()])
        y_pred = torch.cat([y_pred, pred.reshape(-1).cpu().detach()])
    ras = roc_auc_score(y, y_pred)
    return correct / len(loader.dataset), loss_all/len(loader.dataset), ras  # Derive ratio of correct predictions.

def batch_dense_to_sparse(adj):
    edge_index_all = torch.empty([2,0]).long().to(adj.device)
    edge_weight_all = torch.empty([0]).to(adj.device)
    batch_size = adj.shape[0]
    n_nodes = adj.shape[1]
    for i in range(batch_size):
        edge_index_i, edge_weight_i = dense_to_sparse(adj[i])
        edge_index_all = torch.cat([edge_index_all, edge_index_i+n_nodes*i],1)
        edge_weight_all = torch.cat([edge_weight_all, edge_weight_i])
    return edge_index_all, edge_weight_all

def batch_to_dense_adj(edge_index, edge_weight, batch_size, n_nodes):
    adj = torch.empty(batch_size, n_nodes, n_nodes).to(edge_weight.device)
    for i in range(batch_size):
        adj[i] = to_dense_adj(edge_index = edge_index.reshape(2,batch_size,-1).permute(1,0,2)[i]-n_nodes*i, edge_attr=edge_weight.reshape(batch_size,-1)[i])
    return adj

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
