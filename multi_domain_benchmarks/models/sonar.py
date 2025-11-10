
import torch

from torch_geometric.nn import MessagePassing, LayerNorm
from torch_geometric.utils import get_laplacian, add_self_loops, add_remaining_self_loops, remove_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Optional
from helpers.classes import EnvArgs, Pool
from torch.nn import (Module, Dropout, Identity, 
                      Parameter, Linear, Sequential, ReLU, ModuleList, BatchNorm1d, GELU)


class EdgeFeatureAggregator(MessagePassing):
    r"""
    Graph convolution to aggregate the edge_attr features
    Args:
        in_channels (int): The number of input channels.
        use_gradient (bool, optional): Whether to use the gradient operator in message passing (default: :obj:`True`).
    """
    def __init__(self, in_channels, use_gradient: bool = True):
        
        super().__init__(aggr='add')
        self.use_gradient = use_gradient
        
        
    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        ### Copied from the laplacian to get the geometric mean of adjacent degrees
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1),
                                    device=edge_index.device)

        num_nodes = maybe_num_nodes(edge_index)
        row, col = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')
        deg_sqrt = deg.pow_(0.5)
        deg_sqrt.masked_fill_(deg_sqrt == float('inf'), 0)
        # Col and row are inverted to get the transpose
        edge_weight_tr = deg_sqrt[col] * edge_weight * deg_sqrt[row]
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight_tr)
    
    def message(self, x_i, x_j, edge_attr:torch.Tensor, edge_weight:torch.Tensor):
        # Average operator
        avg_op = edge_weight.view((-1,1)) * (x_i + x_j) / 2
        avg_vec = avg_op * edge_attr
        # The gradient operator, keep in mind the pyg notation which is inverted than usual
        grad_op = (x_j - x_i) * edge_weight.view((-1,1))
        grad_vec = grad_op * edge_attr
        return torch.cat([avg_vec, grad_vec], dim=1) if self.use_gradient else avg_vec


class LaplacianAggr(MessagePassing):
    r"""
    Graph convolution which compute the laplacian of the graph with weights based on the edge resistance
    """
    def __init__(self, in_channels, normalization=None):
        """
        Args:
        in_channels (int): The number of input channels.
        normalization (str, optional): The normalization scheme for the graph Laplacian (default: :obj:`None`):

            1. :obj:`None`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"sym"`: Symmetric normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
            \mathbf{D}^{-1/2}`

            3. :obj:`"rw"`: Random-walk normalization
            :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}`
        """
        super().__init__(aggr='add')
        assert normalization in [None, "sym", "rw"]
        self.lin = Linear(in_channels, in_channels, bias=True)
        #self.lin_edge = Linear(in_channels, in_channels, bias=True)
        self.normalization = normalization
        
    def forward(self, x, edge_index=None, edge_resistance=None, edge_attr=None):
        # The original edge_index does not contain self loops
        # Transform the features with a linear map
        in_feature = self.lin(x)
        # Obtaining th laplacian will get the self loops as well
        edge_index_self, edge_resistance_lap = get_laplacian(edge_index=edge_index, edge_weight=edge_resistance, normalization=self.normalization)
        # Perform the message passing
        out = self.propagate(x = in_feature, 
                             edge_index = edge_index_self,
                             edge_resistance = edge_resistance_lap,
                             edge_attr = edge_attr
                             )
        return out

    def message(self, x_j: torch.Tensor, edge_resistance: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        # Since the laplacian contains self-loops this will also get the self information
        if edge_attr is None:   
            return x_j if edge_resistance is None else edge_resistance.view(-1, 1) * x_j
        else:
            return x_j + edge_attr if edge_resistance is None else edge_resistance.view(-1, 1) * (x_j + edge_attr)

    def __repr__(self) -> str:
        return f'self.__class__.__name__(in_channels: {self.in_channels}, normalization: {self.normalization})'


class NullForce(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input):
        return torch.zeros_like(input)


class SONARConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 edge_channels: int,
                 num_iters: int = 1, 
                 epsilon : float = 0.01,
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 normalization: str = None,
                 use_dissipation: bool = False,
                 use_forcing: bool = False,
                 fix_resistance: bool = False,
                 dropout: float = 0.0,
                 bias: bool = False) -> None:

        super().__init__(aggr = 'add')
        self.dropout = Dropout(p=0.) ### BE CAREFUL, this is not the one in the constructor
        
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.num_iters = num_iters
        self.use_dissipation = use_dissipation
        self.use_forcing = use_forcing
        self.epsilon = epsilon
        self.fix_restistance = fix_resistance
        self.bnorm = BatchNorm1d(in_channels)

        self.conv = LaplacianAggr(in_channels, normalization=normalization)

        # Simple net for the dissipative component
        self.dissipative_net = (NullForce() if not self.use_dissipation 
                                 else Sequential(
                                        Linear(in_channels, in_channels),
                                        ReLU(),
                                ))
        # Simple net for the external forcing
        self.external_forcing_net = (NullForce() if not use_forcing 
                                     else Sequential(
                                        Linear(in_channels, in_channels),
                                    ))
        
        # A simple net for the edge resistance
        self.edge_resistance_net = Sequential(
            Linear(in_channels*2, 1), 
            ReLU(),
        )
        
        # A simple net for the initial velocity
        self.velocity_net = Sequential(
            Linear(in_channels, in_channels),
        )
        self.activation = torch.nn.Tanh() if activ_fun != 'Identity' else torch.nn.Identity()
        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Define the velocity vector
        v = self.velocity_net(x)
        
        # Get the edge resistance
        if self.fix_restistance:
            res = (torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1) if edge_attr is not None 
                   else torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
            edge_resistance = self.edge_resistance_net(res).squeeze().abs()
        
        for i in range(self.num_iters):
            # If the edge resistance is not fixed, compute it
            if not self.fix_restistance:
                res = (torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1) if edge_attr is not None 
                       else torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
                edge_resistance = self.edge_resistance_net(res).squeeze().abs()

            # Compute the update term from the laplacian
            conv = self.conv(x, edge_index=edge_index, edge_resistance=edge_resistance, edge_attr=edge_attr)
            # Compute the dissipative term
            diss = self.dissipative_net(x)
            # Compute the external forcing term
            forcing = self.external_forcing_net(x)
            
            # Update the velocity and position of the nodes
            v = v - self.epsilon*(self.dropout(conv) + diss*v - forcing)
            x = x + self.epsilon * v
        return x


#class SONAR(Module):
class BlockSONAR(Module):
    def __init__(self, env_args: EnvArgs, pool: Pool):
        super().__init__()

        self.input_dim = env_args.in_dim
        self.output_dim = env_args.out_dim
        self.hidden_dim = env_args.hid_dim
        self.num_blocks = env_args.num_layers # NOTE in this case we consider the number of layers in standard gnns == to the number of blocks in sonar
        self.num_iters = env_args.num_iters
        self.epsilon = env_args.epsilon
        self.activ_fun = env_args.act_type
        self.bias = True
        #self.train_weights = train_weights
        self.fix_resistance = env_args.fix_resistance
        self.use_dissipation = env_args.use_dissipation
        self.use_forcing = env_args.use_forcing
        self.normalization = env_args.normalization
        self.layer_norm = env_args.layer_norm
        self.activation = env_args.act_type.nn()
        self.aggregate_edge_features = env_args.aggregate_edge_features
        self.residual = env_args.residual
        self.pre_act_norm = env_args.pre_act_norm
        self.resPerNode = False
        
        # Node encoders
        self.use_encoders = env_args.dataset_encoders.use_encoders()
        self.node_encoder = env_args.load_enc()
        self.node_decoder = env_args.load_dec()

        # Edge encoders
        self.dataset_encoder = env_args.dataset_encoders
        self.bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.hid_dim)
        
        
        if self.aggregate_edge_features:
            self.hidden_dim += 2*self.hidden_dim
        
        # Node encoders
        self.use_encoders = env_args.dataset_encoders.use_encoders()
        self.node_encoder = env_args.load_enc()
        self.node_decoder = env_args.load_dec(hid_dim=self.hidden_dim)
    
        params = {
            'in_channels': self.hidden_dim,
            'edge_channels': 0 if self.bond_encoder is None else self.hidden_dim,
            'num_iters': self.num_iters,
            'epsilon': self.epsilon,
            'activ_fun': self.activ_fun,#'Identity', #self.activ_fun,
            'normalization': self.normalization,
            'use_dissipation': self.use_dissipation,
            'use_forcing': self.use_forcing,
            'fix_resistance': self.fix_resistance,
            'bias': self.bias,
            'dropout': env_args.dropout# TODO, add in-conv dropout
        }
        
        self.convs = []
        self.mlps = []
        self.dropouts = []
        self.pre_norms = []
        self.post_norms = []
        for j in range(self.num_blocks):
            self.convs.append(SONARConv(**params))
            self.mlps.append(            
                Sequential(
                    Linear(self.hidden_dim, self.hidden_dim),
                    GELU(),
                    Linear(self.hidden_dim, self.hidden_dim),
                )
            )
            self.pre_norms.append(BatchNorm1d(self.hidden_dim) if env_args.layer_norm else Identity(self.hidden_dim))
            self.dropouts.append(Dropout(p=env_args.dropout))
            self.post_norms.append(BatchNorm1d(self.hidden_dim) if env_args.layer_norm else Identity(self.hidden_dim))
            
            
        self.convs = ModuleList(self.convs)
        self.mlps = ModuleList(self.mlps)
        self.post_norms = ModuleList(self.post_norms)
        self.dropouts = ModuleList(self.dropouts)
        self.pre_norms = ModuleList(self.pre_norms)
        self.initial_layer_norm = BatchNorm1d(self.hidden_dim) if env_args.layer_norm else Identity(self.hidden_dim)
        
        self.drop_ratio = env_args.dropout
        self.pooling = pool.get()
        # Collect the edge feature aggregator, for now is applied only at the beginning
        self.edge_feature_aggregator = EdgeFeatureAggregator(env_args.hid_dim, use_gradient=True)
        print(f'Total parameters: {self.total_parameters()}')
        
    def total_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def forward(self, x, edge_index, pestat, edge_attr = None, batch = None):
        
        # Embed the features, particularly for the molecular datasets
        x = self.node_encoder(x, pestat)
        if edge_attr is None or self.bond_encoder is None:
            edge_embedding = None
        else:
            edge_embedding = self.bond_encoder(edge_attr)
        
        # Aggregate edge features if required
        if self.aggregate_edge_features and edge_embedding is not None:    
            edge_agg = self.edge_feature_aggregator(x, edge_index, edge_attr=edge_embedding)
            x = torch.concat([x, edge_agg], dim=1) if edge_embedding is not None else x
            edge_embedding = None


        if not self.aggregate_edge_features and edge_embedding is not None:
            edge_index, edge_embedding = add_remaining_self_loops(edge_index=edge_index, edge_attr=edge_embedding, fill_value=1., )
        # Initial layer norm
        x = self.initial_layer_norm(x)
        # Pass through the blocks
        for i in range(self.num_blocks): 
            x_in = x
            # SONAR convolution
            x = self.convs[i](x, edge_index, edge_attr=edge_embedding)
            
            # Pre-activation normalization
            if self.pre_act_norm:
                x = self.pre_norms[i](x)
            # Activation and dropout
            x = self.activation(x)
            x = self.dropouts[i](x)
            # Residual connection
            if self.residual:
                x = x + x_in

            # MLP
            x = self.mlps[i](x)
            # Final layer norm
            if self.layer_norm:
                x = self.post_norms[i](x)

        x = self.pooling(x, batch=batch)
        x = self.node_decoder(x)  # decoder
        return x

### Potrebbe essere utile rifare func con questo senza posenc, e provare senza norm