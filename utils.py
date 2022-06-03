import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def train(model, optim, train_loader, val_loader, test_loader, criterion, epochs, wandb, key):
    path = 'models/' + model.__str__()[:3] + str(key) + '.pt'
    best_val_accuracy = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        for data in train_loader:
            if str(model) == 'ShallowConvNet':
                inputs = (data.x.view(-1,61,100).unsqueeze(1).cuda().float().permute(0,1,3,2),)
            elif str(model) == 'EEGNet':
                inputs = (data.x.view(-1,61,100).unsqueeze(1).cuda().float(),)
            else:
                data = data.to('cuda')
                inputs = (data.x.float(), data.edge_index, None, data.batch)

            labels_y = data.y.view(-1,3).float().cuda()

            # zero the parameter gradients
            optim.zero_grad()
            
            out_y, grads = model(*inputs)
            loss_aux = grads.square().mean()
            loss = criterion(out_y, labels_y) + 1e-3*loss_aux# Compute the loss.
            loss.backward()

            optim.step()

            running_loss += loss.item()
            
        running_loss = running_loss / len(train_loader)
        val_acc_s, val_acc_d, val_precision_s, val_precision_d, val_recall_s, val_recall_d, val_f1_s, val_f1_d = test(model, val_loader)
        train_acc_s, train_acc_d, _, _, _, _, _, _ = test(model, train_loader)
        
        if (val_acc_s*val_acc_d) > best_val_accuracy:
            best_val_accuracy = (val_acc_s*val_acc_d)
            try:
                wandb.log({"best val accuracy": best_val_accuracy})
            except:
                None
            torch.save(model.state_dict(), path)
        
        try:
            wandb.log({"epoch": epoch, "running loss": running_loss, 
                       "train accuracy s": train_acc_s, "train accuracy d": train_acc_d,
                       "val precision": (val_precision_s+val_precision_d)/2, 
                       "val recall": (val_recall_s+val_recall_d)/2, "val f1": (val_f1_d + val_f1_s)/2,
                       "val accuracy s": val_acc_s, "val accuracy d": val_acc_d, "val accuracy m": val_acc_s*val_acc_d,
                       "val precision d": val_precision_d, 
                       "val recall d": val_recall_d, "val f1 d": val_f1_d, "val precision s": val_precision_s, 
                       "val recall s": val_recall_s, "val f1 s": val_f1_s})
        except:
            print(f"Epoch: {epoch}, "
              f"Running loss: {running_loss:.2f}, "
              f"Val accuracy: {((val_acc_s+val_acc_d)/2):.2f}, "
              f"Val F1-metric: {((val_f1_d + val_f1_s)/2):.2f}")
            
    model.load_state_dict(torch.load(path))
    model.eval()
    test_acc_s, test_acc_d, test_precision_s, test_precision_d, test_recall_s, test_recall_d, test_f1_s, test_f1_d = test(model, test_loader)
    try:
        wandb.log({"test accuracy": (test_acc_s+test_acc_d)/2,
                   "test precision": (test_precision_s+test_precision_d)/2, 
                   "test recall": (test_recall_s+test_recall_d)/2, "test f1": (test_f1_d + test_f1_s)/2,
                   "test accuracy s": test_acc_s, "test accuracy d": test_acc_d, "test precision d": test_precision_d, 
                   "test recall d": test_recall_d, "test f1 d": test_f1_d, "test precision s": test_precision_s, 
                   "test recall s": test_recall_s, "test f1 s": test_f1_s})
    except:
        print(f"Test accuracy: {((test_acc_s+test_acc_d)/2):.2f}, "
              f"Test F1-metric: {((test_f1_d + test_f1_s)/2):.2f}")

    
def apply_along_axis(function, x, axis: int = 0):
     return torch.stack([
         function(x_i) for x_i in torch.unbind(x, dim=axis)
     ], dim=axis).to(x.device)
        
def test(model, loader):
    model.eval() # set evaluation mode for the model
    correct_s = 0
    correct_d = 0
    precision_s = 0
    precision_d = 0
    recall_s = 0
    recall_d = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        if str(model) == 'ShallowConvNet':
            inputs = (data.x.view(-1,61,100).unsqueeze(1).cuda().permute(0,1,3,2),)
        elif str(model) == 'EEGNet':
            inputs = (data.x.view(-1,61,100).unsqueeze(1).cuda(),)
        else:
            data = data.to('cuda')
            inputs = (data.x.float(), data.edge_index, None, data.batch)
        labels_d, labels_s = data.d.cuda(), data.s.cuda()
        out_y = model(*inputs)
        pred_d = apply_along_axis(func, out_y > 0.5)
        pred_s = (out_y[:,2] > 0.5).long()
        correct_s += int((pred_s == labels_s).sum())  # Check against ground-truth labels.
        correct_d += int((pred_d == labels_d).sum())  # Check against ground-truth labels.
        precision_d += precision_score(labels_d.cpu().detach().numpy(), np.round(pred_d.cpu().detach().numpy()), average='macro')
        precision_s += precision_score(labels_s.cpu().detach().numpy(), np.round(pred_s.cpu().detach().numpy()), average='macro')
        recall_d += recall_score(labels_d.cpu().detach().numpy(), np.round(pred_d.cpu().detach().numpy()), average='macro')
        recall_s += recall_score(labels_s.cpu().detach().numpy(), np.round(pred_s.cpu().detach().numpy()), average='macro')
        
    acc_d = correct_d / len(loader.dataset)
    acc_s = correct_s / len(loader.dataset)
    precision_d = precision_d / len(loader)
    recall_d = recall_d / len(loader)
    f1_d = 2*precision_d*recall_d/ (precision_d+recall_d)
    precision_s = precision_s / len(loader)
    recall_s = recall_s / len(loader)
    f1_s = 2*precision_s*recall_s/ (precision_s+recall_s)
    
    return acc_s, acc_d, precision_s, precision_d, recall_s, recall_d, f1_s, f1_d  # Derive ratio of correct predictions.

