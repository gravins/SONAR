
import torch

from torch.nn import Module, Parameter, Linear, Sequential, LeakyReLU, ReLU, ModuleList
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import get_laplacian
from typing import Optional
#from GraphProp_model_utils import *
from collections import OrderedDict
from torch.func import jacrev

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
        self.lin = Linear(in_channels, in_channels, bias=False)
        self.normalization = normalization
        
    def forward(self, x, edge_index=None, edge_resistance=None):
        # The original edge_index does not contain self loops
        # Transform the features with a linear map
        in_feature = self.lin(x)
        # Obtaining th laplacian will get the self loops as well
        edge_index_self, edge_resistance_lap = get_laplacian(edge_index=edge_index, edge_weight=edge_resistance, normalization=self.normalization)
        out = self.propagate(x = in_feature, 
                             edge_index = edge_index_self,
                             edge_resistance = edge_resistance_lap,
                             )
        return out

    def message(self, x_j: torch.Tensor, edge_resistance: torch.Tensor) -> torch.Tensor:
        # Since the laplacian contains self-loops this will also get the self information
        return x_j if edge_resistance is None else edge_resistance.view(-1, 1) * x_j

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
                 activ_fun: str = 'Identity', # it should be monotonically non-decreasing
                 normalization: str = None,
                 use_dissipation: bool = False,
                 use_forcing: bool = False,
                 fix_resistance: bool = False,
                 bias: bool = False) -> None:

        super().__init__(aggr = 'add')
        self.W = Parameter(torch.empty((in_channels, in_channels)))
        self.bias = Parameter(torch.empty(in_channels)) if bias else None
        
        self.in_channels = in_channels
        self.edge_channels = edge_channels
        self.num_iters = num_iters
        self.use_dissipation = use_dissipation
        self.use_forcing = use_forcing
        self.epsilon = epsilon
        self.fix_restistance = fix_resistance
        
        self.conv = LaplacianAggr(in_channels, normalization=normalization)

        # Simple net for the dissipative component
        self.dissipative_net = (NullForce() if not self.use_dissipation 
                                 else Sequential(
                                        Linear(in_channels, in_channels),
                                        ReLU()
                                ))
        # Simple net for the external forcing
        self.external_forcing_net = (NullForce() if not use_forcing 
                                     else Sequential(
                                        Linear(in_channels, in_channels),
                                        ReLU(),
                                        Linear(in_channels, in_channels)
                                    ))
        
        # A simple net for the edge resistance
        self.edge_resistance_net = Sequential(
            Linear(edge_channels+in_channels*2, in_channels),
            ReLU(),
            Linear(in_channels, 1), # NOTE: The laplacian is defined only with one-dimensional edge weights
        )
        self.activation = getattr(torch, activ_fun) if activ_fun != 'Identity' else torch.nn.Identity()

        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Define the velocity vector, starting with zeros
        # TODO, we could use some encoded thing from the position
        v = torch.zeros_like(x)
        
        res = (torch.cat([x[edge_index[0]], x[edge_index[1]], edge_weight], dim=1) if edge_weight is not None 
               else torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
        edge_resistance = self.edge_resistance_net(res).squeeze().abs()
        for i in range(self.num_iters):
            # If the edge resistance is not fixed, compute it
            if not self.fix_restistance:
                res = (torch.cat([x[edge_index[0]], x[edge_index[1]], edge_weight], dim=1) if edge_weight is not None 
                       else torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
                edge_resistance = self.edge_resistance_net(res).squeeze().abs()

            # Compute the update term from the laplacian
            conv = self.conv(x, edge_index=edge_index, edge_resistance=edge_resistance)
            
            # Compute the dissipative term
            diss = self.dissipative_net(x)
            forcing = self.external_forcing_net(x)
            
            # Update the velocity and position of the nodes
            v = v - self.epsilon*(conv + diss*v - forcing)
            x = x + self.epsilon * v # self.activation(v)
        return x

   


class BlockSONAR(Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_dim,
                 edge_dim,
                 num_blocks,
                 epsilon,
                 num_iters,
                 activ_fun='Tanh',
                 normalization: str = None,
                 node_level_task=False,
                 use_dissipation: bool = True,
                 use_forcing: bool = False,
                 fix_resistance: bool = True,
                 bias: bool = False) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.activ_fun = activ_fun
        self.bias = bias
        self.device = None
        #self.train_weights = train_weights
        self.fix_resistance = fix_resistance
        self.use_dissipation = use_dissipation
        self.use_forcing = use_forcing
        
        self.emb = Linear(self.input_dim, self.hidden_dim)
    
        params = {
            'in_channels': self.hidden_dim,
            'edge_channels': self.edge_dim,
            'num_iters': self.num_iters,
            'epsilon': self.epsilon,
            'activ_fun': 'Identity', #self.activ_fun,
            'normalization': normalization,
            'use_dissipation': self.use_dissipation,
            'use_forcing': self.use_forcing,
            'fix_resistance': self.fix_resistance,
            'bias': self.bias
        }
        
        self.convs = []
        self.mlps = []
        for _ in range(num_blocks):
                self.convs.append(SONARConv(**params))
                self.mlps.append(
                    Sequential(
                        Linear(self.hidden_dim, self.hidden_dim),
                        getattr(torch.nn, activ_fun)(),
                        Linear(self.hidden_dim, self.hidden_dim)
                    )
                )
        self.convs = ModuleList(self.convs)
        self.mlps = ModuleList(self.mlps)
            
        # if not train_weights:
        #     #for param in self.enc.parameters():
        #     #    param.requires_grad = False
        #     for param in self.conv.parameters():
        #         param.requires_grad = False

        self.node_level_task = node_level_task 
        if self.node_level_task:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(self.hidden_dim, self.hidden_dim // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(self.hidden_dim // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(self.hidden_dim * 3, (self.hidden_dim * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((self.hidden_dim * 3) // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))

    def forward(self, data) -> torch.Tensor:
        # Get the data
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None

        # Embed the features
        x = self.emb(x)
        
        # If weight sharing then len(self.conv)==1 else len(self.conv)==num_layers
        for i in range(self.num_blocks):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.mlps[i](x)
        
        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = self.readout(x)
        return x
   