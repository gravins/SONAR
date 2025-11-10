from enum import Enum, auto
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential, Tanh
from torch import Tensor
from typing import NamedTuple, Any, Callable, List
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool, global_add_pool, global_max_pool
import torch
from helpers.metrics import MetricType
from helpers.encoders import DataSetEncoders, PosEncoder, GPPDecoders
from lrgb.encoders.composition import Concat2NodeEncoder


class ActivationType(Enum):
    """
        an object for the different activation types
    """
    RELU = auto()
    GELU = auto()
    TANH = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ActivationType[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is ActivationType.RELU:
            return F.relu
        elif self is ActivationType.GELU:
            return F.gelu
        elif self is ActivationType.TANH:
            return F.tanh
        else:
            raise ValueError(f'ActivationType {self.name} not supported')

    def nn(self) -> Module:
        if self is ActivationType.RELU:
            return ReLU()
        elif self is ActivationType.GELU:
            return GELU()
        elif self is ActivationType.TANH:
            return Tanh()
        else:
            raise ValueError(f'ActivationType {self.name} not supported')



class Pool(Enum):
    """
        an object for the different activation types
    """
    NONE = auto()
    MEAN = auto()
    SUM = auto()
    GPP = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return Pool[s]
        except KeyError:
            raise ValueError()

    def get(self):
        if self is Pool.MEAN:
            return global_mean_pool
        elif self is Pool.SUM:
            return global_add_pool
        elif self is Pool.NONE:
            return BatchIdentity()
        elif self is Pool.GPP:
            return lambda x, batch: torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        else:
            raise ValueError(f'Pool {self.name} not supported')


class EnvArgs(NamedTuple):
    # dimensions
    in_dim: int
    hid_dim: int
    out_dim: int
    #edge_dim: int
    num_layers: int
    num_iters: int

    epsilon: float
    fix_resistance: bool
    use_dissipation: bool
    use_forcing: bool
    normalization: str

    # layer_norm, activation, dropout
    layer_norm: bool
    pre_act_norm: bool
    residual: bool
    dropout: float
    act_type: ActivationType

    # encoder, decoder
    pos_enc: PosEncoder
    dataset_encoders: DataSetEncoders
    dataset_decoders: GPPDecoders
    dec_num_layers: int
    aggregate_edge_features: bool

    def load_enc(self) -> Module:
        if self.pos_enc is PosEncoder.NONE:
            enc_list = self.dataset_encoders.node_encoder(in_dim=self.in_dim, emb_dim=self.hid_dim)
        else:
            if self.dataset_encoders is DataSetEncoders.NONE:
                enc_list = self.pos_enc.get(in_dim=self.in_dim, emb_dim=self.hid_dim)
            else:
                enc_list = Concat2NodeEncoder(enc1_cls=self.dataset_encoders.node_encoder,
                                              enc2_cls=self.pos_enc.get,
                                              in_dim=self.in_dim, emb_dim=self.hid_dim,
                                              enc2_dim_pe=self.pos_enc.DIM_PE())
        return enc_list

    def load_dec(self, hid_dim=None) -> Module:
        if hid_dim is None:
            hid_dim = self.hid_dim
        if self.dataset_decoders is GPPDecoders.NONE:
            if self.dec_num_layers > 1:
                    mlp_list = (self.dec_num_layers - 1) * [Linear(hid_dim, hid_dim),
                                                            Dropout(self.dropout), 
                                                            self.act_type.nn()]
                    mlp_list = mlp_list + [Linear(hid_dim, self.out_dim)]
                    dec_list = Sequential(*mlp_list)
            else:
                dec_list = Linear(hid_dim, self.out_dim)
        else:
            dec_list = self.dataset_decoders.get(self.hid_dim, self.out_dim)
            
        return dec_list


class BatchIdentity(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        return x
