import torch
import torch.nn.functional as F

from torch.nn import Module, Parameter, init, Linear, Sequential
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.nn import inits
from collections import OrderedDict
from typing import Any, Optional
import torch_sparse
import math
import copy



def is_uninitialized_parameter(x: Any) -> bool:
    if not hasattr(torch.nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, torch.nn.parameter.UninitializedParameter)


def reset_weight_(weight: torch.Tensor, in_channels: int,
                  initializer: Optional[str] = None) -> torch.Tensor:
    if in_channels <= 0:
        pass
    elif initializer == 'glorot':
        inits.glorot(weight)
    elif initializer == 'uniform':
        bound = 1.0 / math.sqrt(in_channels)
        torch.nn.init.uniform_(weight.data, -bound, bound)
    elif initializer == 'kaiming_uniform':
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    elif initializer is None:
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")

    return weight


def reset_bias_(bias: Optional[torch.Tensor], in_channels: int,
                initializer: Optional[str] = None) -> Optional[torch.Tensor]:
    if bias is None or in_channels <= 0:
        pass
    elif initializer == 'zeros':
        inits.zeros(bias)
    elif initializer is None:
        inits.uniform(in_channels, bias)
    else:
        raise RuntimeError(f"Bias initializer '{initializer}' not supported")

    return bias

class LinearTransposedSwitching(torch.nn.Module):
    r"""Applies a linear tranformation to the incoming data.

    .. math::
        \mathbf{x}^{\prime} = \mathbf{x} \mathbf{W}^{\top} + \mathbf{b}

    In contrast to :class:`torch.nn.Linear`, it supports lazy initialization
    and customizable weight and bias initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
            If set to :obj:`None`, will match default weight initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
            If set to :obj:`None`, will match default bias initialization of
            :class:`torch.nn.Linear`. (default: :obj:`None`)

    Shapes:
        - **input:** features :math:`(*, F_{in})`
        - **output:** features :math:`(*, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        weight_initializer: Optional[str] = None,
        bias_initializer: Optional[str] = None,
        dtype=torch.float32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        if in_channels > 0:
            self.weight = Parameter(torch.empty(out_channels, in_channels,dtype=dtype))
        else:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)

        if bias:
            self.bias = Parameter(torch.empty(out_channels,dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.transpose_weight = False

        self.reset_parameters()

    def __deepcopy__(self, memo):
        # PyTorch<1.13 cannot handle deep copies of uninitialized parameters :(
        # TODO Drop this code once PyTorch 1.12 is no longer supported.
        out = Linear(
            self.in_channels,
            self.out_channels,
            self.bias is not None,
            self.weight_initializer,
            self.bias_initializer,
        ).to(self.weight.device)

        if self.in_channels > 0:
            out.weight = copy.deepcopy(self.weight, memo)

        if self.bias is not None:
            out.bias = copy.deepcopy(self.bias, memo)

        return out


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset_weight_(self.weight, self.in_channels, self.weight_initializer)
        reset_bias_(self.bias, self.in_channels, self.bias_initializer)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input features.
        """
        if self.transpose_weight:
            message = F.linear(x,self.weight.transpose(0,1),self.bias) 
        else:
            message = F.linear(x,self.weight,self.bias)
        return message


    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if (is_uninitialized_parameter(self.weight)
                or torch.onnx.is_in_onnx_export() or keep_vars):
            destination[prefix + 'weight'] = self.weight
        else:
            destination[prefix + 'weight'] = self.weight.detach()
        if self.bias is not None:
            if torch.onnx.is_in_onnx_export() or keep_vars:
                destination[prefix + 'bias'] = self.bias
            else:
                destination[prefix + 'bias'] = self.bias.detach()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = state_dict.get(prefix + 'weight', None)

        if weight is not None and is_uninitialized_parameter(weight):
            self.in_channels = -1
            self.weight = torch.nn.parameter.UninitializedParameter()
            if not hasattr(self, '_hook'):
                self._hook = self.register_forward_pre_hook(
                    self.initialize_parameters)

        elif weight is not None and is_uninitialized_parameter(self.weight):
            self.in_channels = weight.size(-1)
            self.weight.materialize((self.out_channels, self.in_channels))
            if hasattr(self, '_hook'):
                self._hook.remove()
                delattr(self, '_hook')

        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, bias={self.bias is not None})')

class GCNConvT(GCNConv):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = False,
        dtype = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias,**kwargs)

        self.lin = LinearTransposedSwitching(in_channels, out_channels, bias=False,
                        weight_initializer='glorot',dtype=dtype)
        
    def transpose_mode(self,mode):
        self.lin.transpose_weight = mode

def anti_deriv(x):
    return torch.log(torch.cosh(x))

class SinLayer(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)

class EnergyMLP(Module):
    def __init__(self,in_channels: int,bias: bool = True,activ_fun: str = 'tanh',dtype=torch.float32):
        super().__init__()
        self.W = Parameter(torch.empty((in_channels, in_channels),dtype=dtype))
        self.bias = Parameter(torch.empty(in_channels,dtype=dtype)) if bias else None
        self.activation = getattr(torch, activ_fun)
        self.dtype = dtype
        self.reset_parameters()

    def forward(self, x: torch.Tensor, offset:torch.Tensor) -> torch.Tensor:
        if x.shape[0] != offset.shape[0]:
            print(x.shape[0],offset.shape[0])

        x_non_linear = self.activation(F.linear(x, self.W, (offset + self.bias) if self.bias is not None else offset))
        x = F.linear(x_non_linear,self.W.transpose(0, 1))
        return x,x_non_linear

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

class NaiveConv(MessagePassing):
    def __init__(self,in_channels: int,dtype=torch.float32):
        super().__init__(aggr = 'add')
        self.transpose_weight = False
        self.dtype=dtype
        self.lin = LinearTransposedSwitching(in_channels,in_channels, bias=False,
                        weight_initializer='glorot',dtype=dtype)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor: 
        x_s = x.shape[0]
        message = self.lin(x)
        m_s = message.shape[0]
        aggregation = self.propagate(edge_index=edge_index, x=message, edge_weight=edge_weight)
        a_s = aggregation.shape[0]
        if x_s != m_s or x_s != a_s or m_s != a_s:
            print("conv")
            print(x_s,m_s,a_s)
        return aggregation
    
    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)
    
    def transpose_mode(self,mode):
        self.lin.transpose_weight = mode
    
conv_names = ['naive', 'gcn']
class HamiltonianGradient(MessagePassing):
    def __init__(self,in_channels: int,
                 bias: bool = True,
                 activ_fun: str = 'tanh',
                 conv_mode:int = 0,
                 dtype=torch.float32):
        super().__init__(aggr = 'add')
        assert conv_mode in conv_names, f'{conv_mode} is not in {conv_names}'
        self.conv_mode = conv_mode
        if conv_mode == 'naive':
            self.conv = NaiveConv(in_channels,dtype)
        elif conv_mode == 'gcn':
            self.conv = GCNConvT(in_channels,in_channels,dtype=dtype,bias=False)
            
        self.mlp = EnergyMLP(in_channels,bias,activ_fun,dtype)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor: 
        self.conv.transpose_mode(False) 
        oneHop = self.conv(x,edge_index,edge_weight)
        resnet,non_linear_x = self.mlp(x,oneHop)

        self.conv.transpose_mode(True) 
        twoHop = self.conv(non_linear_x,edge_index,edge_weight)
        return resnet + twoHop
    
    def energy(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.conv.transpose_weight = False
        oneHop = self.conv(x,edge_index)
        self.mlp.activation = anti_deriv
        _,non_linear_x = self.mlp(x,oneHop)
        self.mlp.activation = torch.tanh
        return torch.sum(non_linear_x.flatten())
    
    def get_weights(self):
        if self.conv_mode == 2:
            return self.mlp.W,self.conv.lin_src.weight
        return self.mlp.W,self.conv.lin.weight
    
dampening_names = ['param', 'param+', 'MLP4ReLU', 'DGNReLU']
class InternalDissipation(MessagePassing):
    def __init__(self,in_channels: int, 
                 dampening_mode: str = 'param',
                 dtype = torch.float32):
        super().__init__(aggr = 'add')
        self.nf = in_channels
        self.dampening_mode = dampening_mode
        self.dtype = dtype
        self.relu = torch.nn.ReLU()

        assert self.dampening_mode in dampening_names, f'{self.dampening_mode} is not in {dampening_names}'
        
        if self.dampening_mode == 'param' or self.dampening_mode == 'param+':
            self.dampening = Parameter(torch.zeros(1, in_channels, dtype=self.dtype))
        elif self.dampening_mode == 'MLP4ReLU':
            self.dampening = Sequential(OrderedDict([
                    ('DL1', Linear(in_channels, in_channels,dtype=self.dtype)), 
                    ('DAct1',torch.nn.ReLU()),
                    ('DL2', Linear(in_channels, in_channels,dtype=self.dtype)),
                    ('DAct2',torch.nn.ReLU()),
                    ('DL3', Linear(in_channels, in_channels,dtype=self.dtype)),
                    ('DAct3',torch.nn.ReLU()),
                    ('DL4', Linear(in_channels, in_channels,bias = False,dtype=self.dtype)), #https://github.com/shaandesai1/PortHNN/blob/main/models/TDHNN4.py
                    ('DAct4',torch.nn.ReLU())
                ]))
        elif self.dampening_mode == 'DGNReLU':
            self.dampening = Parameter(torch.empty((in_channels, in_channels),dtype=self.dtype))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor: 
        if self.dampening_mode == 'param':
            damp_factor = self.dampening
        elif self.dampening_mode == 'param+':
            damp_factor =  self.relu(self.dampening)
        elif self.dampening_mode == 'MLP4ReLU':
            damp_factor = self.dampening(x)
        elif self.dampening_mode == 'DGNReLU':
            damp_factor = F.linear(x,self.dampening) 
            damp_factor = self.relu(self.propagate(edge_index, x=damp_factor, edge_weight=edge_weight))
        return damp_factor

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)
    
    def reset_parameters(self):
        if self.dampening_mode == 'DGNReLU':
            inits.glorot(self.dampening)
        if self.dampening_mode == 'param' or self.dampening_mode == 'param+':
            torch.nn.init.kaiming_normal_(self.dampening)

ext_force_names = ['MLP4Sin', 'DGNtanh']
class ExternalForcing(MessagePassing):
    def __init__(self,in_channels: int, 
                 external_mode: str = 'DGNtanh', 
                 dtype = torch.float32, 
                 activ_fun = None):
        super().__init__(aggr = 'add')
        self.nf = in_channels
        self.external_mode = external_mode
        self.dtype = dtype
        self.activation = getattr(torch, activ_fun)

        assert self.external_mode in ext_force_names, f'{self.external_mode} is not in {ext_force_names}'
        if self.external_mode == 'MLP4Sin':
            self.external_force = Sequential(OrderedDict([
                    ('L1', Linear(in_channels + 1, in_channels + 1,dtype=self.dtype)), 
                    ('Sin1',SinLayer()),
                    ('L2', Linear(in_channels + 1, in_channels+1,dtype=self.dtype)),
                    ('Sin2',SinLayer()),
                    ('L3', Linear(in_channels + 1, in_channels+1,dtype=self.dtype)),
                    ('Sin3', SinLayer()),
                    ('L4', Linear(in_channels + 1, in_channels,bias = False,dtype=self.dtype)), #https://github.com/shaandesai1/PortHNN/blob/main/models/TDHNN4.py
                ]))
        elif self.external_mode == 'DGNtanh':
            self.external_force = Parameter(torch.empty((in_channels +1, in_channels),dtype=self.dtype))
        
        self.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None,time_point = None) -> torch.Tensor: 
        time = torch.ones(x.shape[0],1,dtype=self.dtype)*time_point
        time = time.to(x.device)
        state = torch.cat([x,time],dim=1)
        if self.external_mode == 'MLP4Sin':
                external = self.external_force(state)
        elif self.external_mode == 'DGNtanh':
            external = F.linear(state,self.external_force.T) 
            external =  self.activation(self.propagate(edge_index, x=external, edge_weight=edge_weight))
        return external

    def message(self, x_j: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)
    
    def reset_parameters(self):
        if self.external_mode == 2:
            inits.glorot(self.external_force)


class PortHamiltonianConv(MessagePassing):
    def __init__(self, 
                 in_channels: int,
                 num_iters: int = 1, 
                 epsilon : float = 0.1, 
                 activ_fun: str = 'tanh', # it should be monotonically non-decreasing
                 p_conv_mode: str = 'naive',
                 q_conv_mode: str = 'naive',
                 bias: bool = True,
                 beta: float = 0.,
                 alpha : float = 0.,
                 dampening_mode: Optional[str] = None,
                 external_mode: Optional[str] = None,
                 dtype=torch.float32) -> None:
        super().__init__(aggr = 'add')

        self.p_conv_mode = p_conv_mode
        self.q_conv_mode = q_conv_mode

        self.gradient_p = HamiltonianGradient(in_channels//2,bias,activ_fun,p_conv_mode,dtype)   
        self.gradient_q = HamiltonianGradient(in_channels//2,bias,activ_fun,q_conv_mode,dtype)
        #self.device = torch.device("cpu")
        self.bias = bias
        self.activ_fun = activ_fun
        self.dampening_mode = dampening_mode
        self.external_mode = external_mode
        self.alpha = alpha
        self.beta = beta
        if self.dampening_mode is not None:
            self.dampening = InternalDissipation(in_channels//2,
                                                 dampening_mode=dampening_mode,
                                                 dtype=dtype)
        if self.external_mode is not None:
            self.external = ExternalForcing(in_channels//2,
                                            external_mode=external_mode,
                                            dtype=dtype,
                                            activ_fun=activ_fun)
            
        self.nf = in_channels
        self.num_iters = num_iters
        self.epsilon = torch.tensor(epsilon, dtype=dtype)

    def forward(self, x, edge_index, edge_weight=None) -> torch.Tensor: 
        p_n = x[:,:self.nf//2]
        q_n = x[:,self.nf//2:]
        for i in range(self.num_iters):
            if self.alpha == 0 and self.beta == 0:
                q_n = q_n + self.epsilon*self.gradient_p(p_n,edge_index,edge_weight)
                p_n = p_n - self.epsilon*self.gradient_q(q_n,edge_index,edge_weight)
            else:
                grad_q = self.gradient_p(p_n,edge_index,edge_weight)
                q_n = q_n + self.epsilon*grad_q
                p_n = p_n - self.epsilon*self.gradient_q(q_n,edge_index,edge_weight)
                if self.dampening_mode is not None:
                    p_n = p_n - self.epsilon*self.alpha*self.dampening(x=q_n,edge_index=edge_index,edge_weight=edge_weight)*grad_q 
                if self.external_mode is not None:
                    p_n = p_n + self.epsilon*self.beta*self.external(x=q_n,edge_index=edge_index,edge_weight=edge_weight,time_point=self.epsilon*i)

        return torch.cat([p_n,q_n],dim=1)
    
    def energy(self,x:torch.Tensor, edge_index: torch.Tensor):
        p_n = x[:,:self.nf//2]
        q_n = x[:,self.nf//2:]
        return self.gradient_q.energy(q_n,edge_index)+self.gradient_p.energy(p_n,edge_index)
    
    def get_weights(self):
        W_p,V_p = self.gradient_p.get_weights()
        W_q,V_q = self.gradient_q.get_weights()
        zerros = torch.zeros(W_p.shape)
        zerros = zerros.to(self.device)
        W = torch.cat((torch.cat([W_p,zerros],1),torch.cat([zerros.clone(),W_q],1)),0)
        V = torch.cat((torch.cat([V_p,zerros.clone()],1),torch.cat([zerros.clone(),V_q],1)),0)
        return W,V
