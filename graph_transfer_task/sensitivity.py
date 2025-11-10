import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GPSConv
from models.sonar import SONARConv, BlockSONAR_Model
import time
import numpy
import pickle, pandas
import gc
from torch.func import jacrev, vmap

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, conv_name, nlayers, conv_params={}):
        super().__init__()
        self.emb = torch.nn.Linear(input_dim, hidden_dim)
        self.device = None
        self.convs = []
        self.mlps = []
        for _ in range(nlayers):
            if 'sonar' in conv_name:
                self.convs.append(SONARConv(**conv_params))
                self.mlps.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        getattr(torch.nn, conv_params['activ_fun'])(),
                        torch.nn.Linear(hidden_dim, hidden_dim)
                    )
                )
            elif conv_name == 'gcn':
                self.convs.append(GCNConv(**conv_params))
            elif conv_name == 'gps':
                nn = GCNConv(hidden_dim, hidden_dim)
                self.convs.append(GPSConv(hidden_dim, nn, heads=2,#heads=4 if hidden_dim % 4 == 0 else 3 if hidden_dim%3 == 0 else 1,
                                          attn_type='multihead', attn_kwargs={}, # pyg >= 2.4
                                          #attn_dropout=0.0, # pyg < 2.4.0
                                          norm='layer'))
            else:
                raise ValueError(conv_name)
        self.convs = torch.nn.ModuleList(self.convs)
        self.mlps = torch.nn.ModuleList(self.mlps) if len(self.mlps) > 0 else None
        self.dec = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x = self.emb(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.mlps is not None:
                x = self.mlps[i](x)
        x = self.dec(x)
        return x
    
    def forward_sensitivity(self, data):
        x, edge_index = data.x.to(self.device), data.edge_index.to(self.device)
        x = self.emb(x)
        print(x.shape)
        
        print(len(self.convs))
        conv_layer = self.convs[-1]  # Get the last convolutional layer
        
        sensitivity_calc = jacrev(lambda x: conv_layer.forward(x, edge_index))
        #sensitivity = vmap(sensitivity_calc)(x)
        sensitivity = sensitivity_calc(x)
        return sensitivity


#############################################

device = torch.device('cuda:2')
import os
from graph_transfer_data import GraphTransferDataset
exp_list = []

data_path = os.path.join('.', 'data')
# Create dataset if it does not exist
pre_transform = None
tr = torch.load(os.path.join(data_path, 'line_50/pre_transform_None/train_line_50_pre_transform_None.pt'), weights_only=False)
data = tr[:1][0]



from graph_transfer_data import ring_transfer_graph, line_graph
data = ring_transfer_graph(distance=50, channels=1, add_crosses=False)
data = line_graph(distance=50, channels=1)
from torch_geometric.utils import remove_self_loops
data.edge_index = remove_self_loops(data.edge_index)[0]
print(data)
hidden_dim = 10
params = {
    'in_channels': hidden_dim,
    'edge_channels': 0,
    'num_iters': 50,
    'epsilon': 0.15,
    'activ_fun': 'Tanh',
    'normalization': 'rw',
    'use_dissipation': False,
    'use_forcing': False,
    'fix_resistance': False,
    'bias': True
}

num_iters_list = [25, 50, 75, 100]
sensitivity_results = {}

for num_iters in num_iters_list:
    print(f'Running sensitivity for num_iters={num_iters}')
    model = BlockSONAR_Model(
        in_channels=1,
        out_channels=1,
        hidden_channels=10,
        num_blocks=1,
        edge_dim=0,
        activ_fun='Identity',
        num_iters=num_iters,
        epsilon=0.15,
        normalization='rw',
        use_dissipation=False,
        use_forcing=False,
        fix_resistance=True
    )
    model.to(device)
    model.device = device
    
    sensitivity = torch.zeros((data.x.shape[0], hidden_dim, hidden_dim), device=device)
    eps = 1e-2
    emb_out = model.emb(data.x.to(device))
    for j in range(sensitivity.shape[2]):
        x_plus = emb_out.clone()
        x_plus[0, j] += eps
        out_plus = model.convs[-1](x_plus, data.edge_index.to(device))

        x_minus = emb_out.clone()
        x_minus[0, j] -= eps
        out_minus = model.convs[-1](x_minus, data.edge_index.to(device))

        sensitivity[:, :, j] = (out_plus - out_minus) / (2 * eps)

    # Save sensitivity results in a specific directory
    save_dir = 'sensitivity_results'
    os.makedirs(save_dir, exist_ok=True)
    sensitivity_results[num_iters] = sensitivity.cpu().detach().numpy()
    numpy.save(os.path.join(save_dir, f'sensitivity_matrix_numiters{num_iters}.npy'), sensitivity_results[num_iters])
    print(f'Sensitivity shape for num_iters={num_iters}:', sensitivity.shape)
    print(sensitivity[0, :, :])
    print(torch.linalg.norm(sensitivity[-1, :, :], ord=1))



for num_iters in num_iters_list:
    print(f'Running sensitivity for num_iters={num_iters} (GNN standard model)')
    model = GNN(
        input_dim=1,
        output_dim=1,
        hidden_dim=hidden_dim,
        conv_name='gcn',  # or 'gps', 'sonar', etc. as needed
        nlayers=num_iters,
        conv_params={'in_channels': hidden_dim, 'out_channels': hidden_dim}
    )
    model.to(device)
    model.device = device

    sensitivity = torch.zeros((data.x.shape[0], hidden_dim, hidden_dim), device=device)
    eps = 1e-2
    emb_out = model.emb(data.x.to(device))
    for j in range(sensitivity.shape[2]):
        x_plus = emb_out.clone()
        x_plus[0, j] += eps
        out_plus = model.convs[-1](x_plus, data.edge_index.to(device))

        x_minus = emb_out.clone()
        x_minus[0, j] -= eps
        out_minus = model.convs[-1](x_minus, data.edge_index.to(device))

        sensitivity[:, :, j] = (out_plus - out_minus) / (2 * eps)

    # Save sensitivity results in a specific directory
    save_dir = 'sensitivity_results_gnn'
    os.makedirs(save_dir, exist_ok=True)
    sensitivity_results[num_iters] = sensitivity.cpu().detach().numpy()
    numpy.save(os.path.join(save_dir, f'sensitivity_matrix_gnn_numiters{num_iters}.npy'), sensitivity_results[num_iters])
    print(f'Sensitivity shape for num_iters={num_iters}:', sensitivity.shape)
    print(sensitivity[0, :, :])
    print(torch.linalg.norm(sensitivity[-1, :, :], ord=1))
    

# Now load the results in a pandas DataFrame
import pandas as pd

sensitivity_matrices = []
sensitivity_matrices_gnn = []
for num_iters in num_iters_list:
    sensitivity_matrix = numpy.load(os.path.join('sensitivity_results', f'sensitivity_matrix_numiters{num_iters}.npy'))
    sensitivity_matrix_gnn = numpy.load(os.path.join('sensitivity_results_gnn', f'sensitivity_matrix_gnn_numiters{num_iters}.npy'))
    # Join the two DataFrames
    sensitivity_matrices.append(sensitivity_matrix)
    sensitivity_matrices_gnn.append(sensitivity_matrix_gnn)

distances = [0, 10, 20, 30, 40, 49]
for distance in distances:
    for i, num_iters in enumerate(num_iters_list):
        sensitivity_norm = numpy.linalg.norm(sensitivity_matrices[i][distance, :, :], ord=1)
        sensitivity_norm_gnn = numpy.linalg.norm(sensitivity_matrices_gnn[i][distance, :, :], ord=1)
        print(f'Num iters: {num_iters}, Distance: {distance}, Sensitivity norm: {sensitivity_norm}, Sensitivity norm GNN: {sensitivity_norm_gnn}')


raise ValueError('This is a test script, not meant to be run directly.')


