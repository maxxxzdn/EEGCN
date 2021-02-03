import torch
import torch.nn.functional as F

"""
Functions to train and estimate a model
"""


def my_loss(x_predicted, x_exact, u_predicted, u_exact):
    l1 = F.cross_entropy(u_predicted, u_exact)
    l2 = F.mse_loss(x_predicted, x_exact)
    return l1 + l2


def train(model, optimizer, train_dataset, device):
    model.train()  # set training mode for the model

    loss_all = 0
    for pair in train_dataset:  # Iterate in batches over the training dataset.
        current, next_ = pair
        current = current.to(device)
        next_ = next_.to(device)
        # Perform a single forward pass.
        update_x, diagnosis = model(
            current.x, current.edge_index, current.u, current.batch)
        x_predicted = update_x + model.encoder(current.x)
        x_exact = model.encoder(next_.x)
        # Compute the loss.
        loss = my_loss(x_predicted, x_exact, diagnosis, current.y)
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
        update_x, diagnosis = model(
            current.x, current.edge_index, current.u, current.batch)
        x_predicted = update_x + model.encoder(current.x)
        x_exact = model.encoder(next_.x)
        pred = diagnosis.argmax(dim=1)
        correct += int(pred == next_.y)
        MSE += F.mse_loss(x_predicted, x_exact)
    return MSE, correct / len(dataset)  # Derive ratio of correct predictions.
