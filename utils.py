import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx

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
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
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
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

"""
Functions to calculate node importance using Integrated Gradients [captum]
"""

# Main function that returns graph visualization with node importances depicted with node color and size    
def explain(model, dataset, positions, labels, names):
   # labels presented in dataset
    nEdges = dataset[0].edge_index.shape[1]
    ultimate_mask = np.zeros((len(names), nEdges)) # 3 x number of edges 
    edge_mask_dict = [] 
    for data in dataset:
        label = data.y.item() 
        edge_mask = explain_edges(model = model, data = data, target=label) # calculate edge importances for the given graph
        ultimate_mask[label] += edge_mask # summarize importances of edges across all the data
    for label in range(len(names)): # for each node aggregate importances of ongoing edges
        edge_mask_dict.append(aggregate_edge_directions(ultimate_mask[label], data.edge_index))
    
    # upload info about graph for nice plot                  
    data.pos = positions 
    data.labels = labels
    G = to_networkx(data, to_undirected=True)
         
    for label in range(len(names)): # for each label visualize result and save it in home directory
        for i in range(nEdges):
            if max(edge_mask_dict[label].values()) > 0:
                edge_mask_dict[label][i] = edge_mask_dict[label][i]/max(edge_mask_dict[label].values()) 
        visualize_edge(G, data, edge_mask_dict[label], 10, 
                 'Signal: ' + str(names[label]) + '; Node importance; method: Integrated Gradients')       
        plt.savefig(str(label) + ".png")
    return edge_mask_dict
        
# To modify forward function for using with captum IntegratedGradients       
def model_forward(edge_mask, data, model):
    batch = torch.zeros(data.x.shape[0], dtype=int) # batch for one single graph
    out = model(data.x, data.edge_index, batch, edge_mask) 
    return out 

# To calculate edge importances for a given graph
def explain_edges(model, data, target=0):
    input_mask = torch.ones(data.edge_index.shape[1], requires_grad = True) # weight = 1 for each edge 
    ig = IntegratedGradients(model_forward) 
    mask = ig.attribute(input_mask, target=target, # edge weights as input to calculate attribute for
                            additional_forward_args=(data,model),
                            internal_batch_size=data.edge_index.shape[1])
    edge_mask = np.abs(mask.cpu().detach().numpy()) 
    return edge_mask
    
# To aggregate importance for each edge from its both directions          
def aggregate_edge_directions(edge_mask, edge_index):
    edge_mask_dict = defaultdict(float) # dictionary storing importance for each edges
    for val, u, v in list(zip(edge_mask, *edge_index)):
        u, v = u.item(), v.item() # vertex indices
        if u > v:
            u, v = v, u # to make it undirected
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict

"""
Functions to visualize graph using networkx
""" 
    
# Convert list of positions to dictionary for networkx
def pos_to_dict(positions):
    position_dict = dict()
    for i in range(len(positions)):
        position_dict[i] = np.array(positions[i])
    return position_dict
    
# Convert list of node labels to dictionary for networkx
def labels_to_dict(labels):
    labels_dict = dict()
    for i in range(len(labels)):
        labels_dict[i] = np.array(labels[i])
    return labels_dict
    
# Main function to visualize a graph
# Input: networkx.Graph, graph data, node importance, size coefficient to tune node size, title of the plot
# Output: graph visualization (matplotlib.pyplot)  
        
def visualize_edge(h, data, edge_mask = None, coefSize = 1000, title = None):
    plt.figure(figsize=(14,14))
    plt.xticks([])
    plt.yticks([])
    if edge_mask is None:
        edge_size = 0
        widths = None
    else: # convert dictionary to lists
        edge_width = [coefSize*edge_mask[(u, v)]/max(edge_mask.values()) for u, v in h.edges()] 
        edge_color = [edge_mask[(u, v)]/max(edge_mask.values()) for u, v in h.edges()]
    if data.pos is not None:
        pos = pos_to_dict(data.pos) # node positions if given
    else: 
        pos = nx.spring_layout(h, seed=42) # automatic calculation of node positions otherwise
    if data.labels is not None:
        labels = labels_to_dict(data.labels) # labels of nodes if given
    else: 
        labels = None # Just numbers otherwise
    nx.draw_networkx(h, pos=pos, labels = labels, width=edge_width, node_size = 0, edge_color = edge_color, edge_cmap=plt.cm.Blues, cmap = 'Set2')
    if title:
        plt.title(title)
