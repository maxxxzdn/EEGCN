import torch
import torch.nn.functional as F

"""
Functions to train and estimate a model
"""


def my_loss(x_predicted, x_exact): #, u_predicted, u_exact):
    #l1 = F.cross_entropy(u_predicted, u_exact)
    l2 = F.mse_loss(x_predicted, x_exact)
    return l2 #l1 + l2


def train(model, optimizer, train_dataset, device):
    model.train()  # set training mode for the model
    model.encoder.eval()
    loss_all = 0
    for pair in train_dataset:  # Iterate in batches over the training dataset.
        current, next_ = pair
        current = current.to(device)
        next_ = next_.to(device)
        # Perform a single forward pass.
        update_x = model(
            next_.x, next_.edge_index, next_.u, next_.batch)
        x_predicted = update_x + model.encoder(next_.x).detach()
        x_exact = model.encoder(current.x).detach()
        # Compute the loss.
        loss = my_loss(x_predicted, x_exact) #, diagnosis, current.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss_all += loss.item()
    return loss_all


def test(model, dataset, device):
    model.eval()  # set evaluation mode for the model

    correct = 0
    MSE = 0

    for pair in dataset:  # Iterate in batches over the training dataset.
        current, next_ = pair
        current = current.to(device)
        next_ = next_.to(device)
        # Perform a single forward pass.
        update_x = model(
            next_.x, next_.edge_index, next_.u, next_.batch)
        x_predicted = update_x + model.encoder(next_.x)
        x_exact = model.encoder(current.x)
        #pred = diagnosis.argmax(dim=1)
        #correct += int(pred == next_.y)
        MSE += F.mse_loss(x_predicted, x_exact)
    return MSE #, correct / len(dataset)  # Derive ratio of correct predictions.

def cl_train(model, optimizer, criterion, train_loader, epoch, device):
    model.train() # set training mode for the model
    
    if epoch % 100 == 0 and optimizer.param_groups[0]['lr'] > 1e-6:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
            
    loss_all = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        loss_all += loss.item()       
    return loss_all

def cl_test(model, loader, device):
    model.eval() # set evaluation mode for the model

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.
