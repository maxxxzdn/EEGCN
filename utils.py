import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import to_networkx

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
def explain(model, dataset, positions, labels):
    names = ['None', 'Right', 'Left'] # labels presented in dataset
    ultimate_mask = np.zeros((3, 184)) # 3 x number of edges 
    node_mask_dict = [] 
    for data in dataset:
        label = data.y.item() 
        edge_mask = explain_edges(model = model, data = data, target=label) # calculate edge importances for the given graph
        ultimate_mask[label] += edge_mask # summarize importances of edges across all the data
    for label in range(3): # for each node aggregate importances of ongoing edges
        node_mask_dict.append(aggregate_node_info(ultimate_mask[label], data.edge_index))
    
    # upload info about graph for nice plot                  
    data.pos = positions 
    data.labels = labels	 
    G = to_networkx(data, to_undirected=True)
         
    for label in range(3): # for each label visualize result and save it in home directory
        visualize_node(G, data, node_mask_dict[label], 100, 
                 'Signal: ' + str(names[label]) + '; Node importance; method: Integrated Gradients')       
        plt.savefig(str(label) + ".png")
        
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

# To aggregate importance for each node from its outgoing edges           
def aggregate_node_info(node_mask, edge_index):
    node_mask_dict = defaultdict(float) # dictionary storing importance for each node
    num_edges = defaultdict(float) # dictionary storing amount of outgoing edges for each node
    for val, u, v in list(zip(node_mask, *edge_index)): 
        u = u.item()
        node_mask_dict[u] += val # add importance of outgoing edge
        num_edges[u] += 1 # one more outgoing edge
    for u in range(61):
        node_mask_dict[u] = node_mask_dict[u]/num_edges[u] # calculate mean
    return node_mask_dict
    
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
def visualize_node(h, data, node_mask = None, coefSize = 250, title = None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    if node_mask is None:
        node_size = 0
        widths = None
    else: # convert dictionary to lists
        node_size = [coefSize*node_mask[u]/max(node_mask.values()) for u in h.nodes()] 
        node_color = [node_mask[u]/max(node_mask.values()) for u in h.nodes()]
    if data.pos is not None:
        pos = pos_to_dict(data.pos) # node positions if given
    else: 
        pos = nx.spring_layout(h, seed=42) # automatic calculation of node positions otherwise
    if data.labels is not None:
        labels = labels_to_dict(data.labels) # labels of nodes if given
    else: 
        labels = None # Just numbers otherwise
    nx.draw_networkx(h, pos=pos, labels = labels, node_size = node_size, node_color = node_color, cmap="Reds")
    if title:
        plt.title(title)
