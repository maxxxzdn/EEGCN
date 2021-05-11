import torch
import numpy as np
"""
Functions to train and estimate a model
"""
def train(model, optimizer, criterion, train_loader, epoch, device):
    model.train() # set training mode for the model
    
    if epoch % 100 == 0 and optimizer.param_groups[0]['lr'] > 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
            
    loss_all = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(x = data.x, edge_index = data.edge_index, batch = data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss_all += loss.item()       
    return loss_all

def test(model, loader, device):
    model.eval() # set evaluation mode for the model

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(x = data.x, edge_index = data.edge_index, batch = data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return torch.diag((A_mA@B_mB.T) / torch.sqrt((ssA[:, None]@ssB[None])))

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == torch.nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)