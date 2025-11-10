from enum import Enum, auto
from torch.nn import Linear, LeakyReLU, Sequential
from torch import Tensor
from torch_geometric.data import Data
from collections import OrderedDict


from lrgb.encoders.laplace import LapPENodeEncoder, LAP_DIM_PE
from lrgb.encoders.kernel import RWSENodeEncoder, KER_DIM_PE
from lrgb.encoders.mol_encoder import AtomEncoder, BondEncoder
from lrgb.encoders.voc_encoder import VOCEdgeEncoder, VOCNodeEncoder


class EncoderLinear(Linear):
    def forward(self, x: Tensor, pestat=None) -> Tensor:
        return super().forward(x)


class DataSetEncoders(Enum):
    """
        an object for the different encoders
    """
    NONE = auto()
    MOL = auto()
    VOC = auto()  # Pascal VOC dataset encoders

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetEncoders[s]
        except KeyError:
            raise ValueError()

    def node_encoder(self, in_dim: int, emb_dim: int):
        if self is DataSetEncoders.NONE:
            return EncoderLinear(in_features=in_dim, out_features=emb_dim)
        elif self is DataSetEncoders.MOL:
            return AtomEncoder(emb_dim)
        elif self is DataSetEncoders.VOC:
            return VOCNodeEncoder(emb_dim)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def edge_encoder(self, emb_dim: int):
        if self is DataSetEncoders.NONE:
            return None
        elif self is DataSetEncoders.MOL:
            return BondEncoder(emb_dim)
        elif self is DataSetEncoders.VOC:
            return VOCEdgeEncoder(emb_dim)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def use_encoders(self) -> bool:
        return self is not DataSetEncoders.NONE








class PosEncoder(Enum):
    """
        an object for the different encoders
    """
    NONE = auto()
    LAP = auto()
    RWSE = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return PosEncoder[s]
        except KeyError:
            raise ValueError()

    def get(self, in_dim: int, emb_dim: int, expand_x: bool = False):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return LapPENodeEncoder(dim_in=in_dim, dim_emb=emb_dim, expand_x=expand_x)
        elif self is PosEncoder.RWSE:
            return RWSENodeEncoder(dim_in=in_dim, dim_emb=emb_dim, expand_x=expand_x)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def DIM_PE(self):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return LAP_DIM_PE
        elif self is PosEncoder.RWSE:
            return KER_DIM_PE
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')

    def get_pe(self, data: Data, device):
        if self is PosEncoder.NONE:
            return None
        elif self is PosEncoder.LAP:
            return [data.EigVals.to(device), data.EigVecs.to(device)]
        elif self is PosEncoder.RWSE:
            return data.pestat_RWSE.to(device)
        else:
            raise ValueError(f'DataSetEncoders {self.name} not supported')


class GPPDecoders(Enum):
    """
        an object for the different deconders in the graph property prediction experiment
    """
    NONE = auto()
    NODE = auto() # graph property prediction - node level prediction
    GRAPH = auto() # graph property prediction - graph level prediction

    @staticmethod
    def from_string(s: str):
        try:
            return GPPDecoders[s]
        except KeyError:
            raise ValueError()

    def get(self, emb_dim: int, out_dim: int):
        if self is GPPDecoders.NONE:
            return None
        elif self is GPPDecoders.NODE:
            return Sequential(OrderedDict([
                ('L1', Linear(emb_dim, emb_dim // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(emb_dim // 2, out_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        elif self is GPPDecoders.GRAPH:
            return Sequential(OrderedDict([
                ('L1', Linear(emb_dim * 3, (emb_dim * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((emb_dim * 3) // 2, out_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            raise ValueError(f'GPPDecoders {self.name} not supported')
