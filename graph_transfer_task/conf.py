from models import *
from utils import cartesian_product
import numpy as np



def get_PHDGN_conservative_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers':[1],
        'activation':['tanh'],
        'num_iters': [distance * ni for ni in range(1,4)],
        'double_dim': [False, True],
        'pq': ['p', 'q', 'pq'],
        'epsilon': [0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 1e-4],
        'p_conv_mode': ['naive', 'gcn'],
        'q_conv_mode': ['naive', 'gcn']
    }
    for params in cartesian_product(grid):
        if params['p_conv_mode']=='gcn' and params['q_conv_mode']=='naive':
            continue
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_PHDGN_conf(in_channels, distance):
    grid = {
        'alpha': [0., 1.], # no external force/dampening
        'beta': [0., 1.], # no external force/dampening
        'dampening_mode': ['param'],
        'external_mode': ['DGNtanh']
    }
    for params in get_PHDGN_conservative_conf(in_channels, distance):
        for conf in cartesian_product(grid):
            if conf['alpha'] == 0 and conf['beta'] == 0: 
                continue ## in this case we are back to conservative PHDGN
            params.update(conf)
            yield params
    

def get_GNN_conf(in_channels, distance):
    # GCN, GAT, GraphSAGE, GIN, GPS
    grid = {
        'hidden_channels':[64],
        'num_layers': [distance],
        'activation':['tanh'],
    }
    for params in cartesian_product(grid):
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_ADGN_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers': [1],
        'activation':['tanh'],
        'num_iters': [distance],
        'gamma': [0.1, 0.2],
        'epsilon': [0.6, 0.2, 0.5, 0.1],
        'graph_conv': ['NaiveAggr', 'GCNConv']
    }
    for params in cartesian_product(grid):
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_SWAN_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers': [1],
        'activation':['tanh'],
        'num_iters': [distance],
        'gamma': [0.1, 0.2],
        'epsilon': [0.6, 0.2, 0.5, 0.1],
        'beta': [1., -1., 0.1, 0.01], #0.01, 0]:,
        'graph_conv': ['AntiSymNaiveAggr', 'BoundedGCNConv', 'BoundedNaiveAggr'],
        'attention': [False, True]
    }
    for params in cartesian_product(grid):
        if params['attn'] and params['conv'] == 'BoundedGCNConv':
            continue
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_SONAR_conf(in_channels, distance):
    grid = {
        'hidden_channels':[64],
        'num_layers': [1],
        'activation':['Tanh'],
        'num_iters': [distance * ni for ni in range(1,3)],
        'epsilon': [1., 0.5, 0.2, 0.1, 0.05, 1e-2],
        'normalization': [None, 'sym', 'rw'],
        'use_dissipation': [True, False],
        'use_forcing': [True, False],
        'fix_resistance': [True, False],
    }
    for params in cartesian_product(grid):
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params


def get_BlockSONAR_conf(in_channels, distance):
    # Second grid
    grid = {
        'hidden_channels':[64],
        'num_blocks': [1, 2],
        'edge_dim': [0],
        'activ_fun':['Tanh'],
        'num_iters': [distance * ni for ni in range(1,3)],
        'epsilon': [1., 0.5, 0.1, 0.05],
        'normalization': [None],
        'use_dissipation': [True, False],
        'use_forcing': [True, False],
        'fix_resistance': [True, False],
    }
    for params in cartesian_product(grid):
        if params['num_blocks'] == 2:
            params['num_iters'] = (params['num_iters']//2 + 1)
        params['in_channels'] = in_channels
        params['out_channels'] = in_channels
        yield params

MODELS = {
    'gin' : (GIN_Model, get_GNN_conf),
    'gcn' : (GCN_Model, get_GNN_conf),
    'gat' : (GAT_Model, get_GNN_conf),
    'sage': (SAGE_Model, get_GNN_conf),
    'swan': (SWAN_Model, get_SWAN_conf),
    'adgn': (ADGN_Model, get_ADGN_conf),
    'phdgn': (PHDGN_Model, get_PHDGN_conf),
    'phdgn_conservative': (PHDGN_Model, get_PHDGN_conservative_conf),
    'gps': (GPS_Model, get_GNN_conf),
    #'sonar': (SONAR_Model, get_SONAR_conf)
    'blocksonar': (BlockSONAR_Model, get_BlockSONAR_conf)
}