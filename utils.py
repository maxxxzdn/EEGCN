import torch.nn.functional as F

# Functions to train and estimate the Euler model


def train(model, optimizer, dataset, device):
    """Function to train the Euler model on a dataset.
    It trains the model on each pair of consequent states from the dataset one single time.
    For each pair it calculates the loss and performs the optimizer step.

    Args:
        model: An instance of Pipeline class.
        optimizer: Pytorch torch.optim algorithm.
        dataset: List of lists containing two torch_geometric.data.Data objects
        device: torch.device object

    Returns:
        loss value calculated on the whole dataset.
    """

    model.train()
    model.encoder.eval()  # Make the model encoder deterministic

    loss_all = 0
    for pair in dataset:
        current, next_ = pair
        current = current.to(device)
        next_ = next_.to(device)

        update_x = model(
            next_.x, next_.edge_index, next_.u, next_.batch)
        x_predicted = update_x + model.encoder(next_.x).detach()
        x_exact = model.encoder(current.x).detach()

        loss = F.mse_loss(x_predicted, x_exact)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_all += loss.item()
    return loss_all


def test(model, dataset, device):
    """Function to test a model on a dataset.
    It calculates mean square error between estimated current state and correct one.

    Args:
        model: An instance of Pipeline class.
        dataset: List of lists containing two torch_geometric.data.Data objects
        device: torch.device object

    Returns:
        Total MSE: sum of MSE values calculated for each pair of consequent states in the dataset.
    """
    model.eval()

    mse = 0
    for pair in dataset:
        current, next_ = pair
        current = current.to(device)
        next_ = next_.to(device)

        update_x = model(
            next_.x, next_.edge_index, next_.u, next_.batch)
        x_predicted = update_x + model.encoder(next_.x)
        x_exact = model.encoder(current.x)
        mse += F.mse_loss(x_predicted, x_exact)

    return mse

# Functions to train and estimate a classifier for the Euler model


def classifier_train(model, optimizer, criterion, loader, device):
    """Function to train a classification model on a loader.
    It trains the model on mini-batches provided by the loader.
    For each mini-batch it calculates the loss and performs the optimizer step.

    Args:
        model: classifier attribute of an instance of Pipeline class.
        optimizer: Pytorch torch.optim algorithm.
        criterion: torch.nn criterion
        loader: torch_geometric.data.DataLoader object containing graphs
        device: torch.device object

    Returns:
        loss value calculated on the whole loader.
    """
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_all += loss.item()
    return loss_all


def classifier_test(model, loader, device):
    """Function to test a classification model on a loader.
    It calculates accuracy of the model prediction on the whole loader .

    Args:
        model: An instance of Pipeline class.
        loader: torch_geometric.data.DataLoader object containing graphs
        device: torch.device object

    Returns:
        percentage of correct predictions
    """
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)

        out = model(data.x, data.edge_index, data.batch)

        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)
