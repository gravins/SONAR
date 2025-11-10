import os
import copy
import os.path as osp
from enum import Enum, auto
import torch
from torch import Tensor
from typing import NamedTuple, Optional, List, Union, Callable
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset
import json
import numpy as np
from torch_geometric.datasets import LRGBDataset

from helpers.dataset_classes.root_neighbours_dataset import RootNeighboursDataset
from helpers.dataset_classes.cycles_dataset import CyclesDataset
from helpers.dataset_classes.lrgb import PeptidesFunctionalDataset, PeptidesStructuralDataset
from helpers.dataset_classes.classic_datasets import Planetoid
#from helpers.constants import ROOT_DIR
from helpers.metrics import MetricType
from helpers.classes import ActivationType, Pool
from helpers.encoders import DataSetEncoders, PosEncoder, GPPDecoders
from lrgb.cosine_scheduler import cosine_with_warmup_scheduler
from lrgb.transforms import apply_transform
from helpers.dataset_classes.graph_prop_pred_dataset import GPPDataset

class DatasetBySplit(NamedTuple):
    train: Union[Data, List[Data]]
    val: Union[Data, List[Data]]
    test: Union[Data, List[Data]]


class DataSetFamily(Enum):
    heterophilic = auto()
    synthetic = auto()
    social_networks = auto()
    proteins = auto()
    lrgb = auto()
    homophilic = auto()
    gpp = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSetFamily[s]
        except KeyError:
            raise ValueError()


class DataSet(Enum):
    """
        an object for the different datasets
    """
    # heterophilic
    roman_empire = auto()
    amazon_ratings = auto()
    minesweeper = auto()
    tolokers = auto()
    questions = auto()

    # synthetic
    root_neighbours = auto()
    cycles = auto()

    # social networks
    imdb_binary = auto()
    imdb_multi = auto()
    reddit_binary = auto()
    reddit_multi = auto()
    
    # proteins
    enzymes = auto()
    proteins = auto()
    nci1 = auto()

    # lrgb
    func = auto()
    struct = auto()
    pascal = auto()

    # homophilic
    cora = auto()
    pubmed = auto()

    # graph property prediction
    eccentricity = auto()
    sssp = auto()
    diameter = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return DataSet[s]
        except KeyError:
            raise ValueError()
        
    def get_family(self) -> DataSetFamily:
        if self in [DataSet.roman_empire, DataSet.amazon_ratings, DataSet.minesweeper,
                    DataSet.tolokers, DataSet.questions]:
            return DataSetFamily.heterophilic
        elif self in [DataSet.root_neighbours, DataSet.cycles]:
            return DataSetFamily.synthetic
        elif self in [DataSet.imdb_binary, DataSet.imdb_multi, DataSet.reddit_binary, DataSet.reddit_multi]:
            return DataSetFamily.social_networks
        elif self in [DataSet.enzymes, DataSet.proteins, DataSet.nci1]:
            return DataSetFamily.proteins
        elif self in [DataSet.func, DataSet.struct, DataSet.pascal]:
            return DataSetFamily.lrgb
        elif self in [DataSet.cora, DataSet.pubmed]:
            return DataSetFamily.homophilic
        elif self in [DataSet.eccentricity, DataSet.diameter, DataSet.sssp]:
            return DataSetFamily.gpp
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def is_node_based(self) -> bool:
        return self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.homophilic]\
            or self is DataSet.root_neighbours or self in [DataSet.eccentricity, DataSet.sssp]\
            or self is DataSet.pascal

    def not_synthetic(self) -> bool:
        return self.get_family() is not DataSetFamily.synthetic

    def is_expressivity(self) -> bool:
        return self is DataSet.cycles

    def clip_grad(self) -> bool:
        return self.get_family() is DataSetFamily.lrgb

    def get_dataset_encoders(self):
        if self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.synthetic, DataSetFamily.social_networks,
                                 DataSetFamily.proteins, DataSetFamily.homophilic, DataSetFamily.gpp]:
            return DataSetEncoders.NONE
        elif self is DataSet.func or self is DataSet.struct:
            return DataSetEncoders.MOL
        elif self is DataSet.pascal:
            return DataSetEncoders.VOC
        else:
            raise ValueError(f'DataSet {self.name} not supported in get_dataset_encoders')
    
    def get_dataset_decoders(self):
        if self.get_family() is DataSetFamily.gpp:
            if self.is_node_based():
                return GPPDecoders.NODE
            else:
                return GPPDecoders.GRAPH
        else:
            return GPPDecoders.NONE

    def get_folds(self, fold: int) -> List[int]:
        if self.get_family() in [DataSetFamily.synthetic, DataSetFamily.lrgb, DataSetFamily.gpp]:
            return list(range(1))
        elif self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.homophilic]:
            return list(range(10))
        elif self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins]:
            return [fold]
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')
    
    def load(self, root_dir:str, seed: int, pos_enc: PosEncoder) -> List[Data]:
        root = osp.join(root_dir, 'datasets')
        os.makedirs(root, exist_ok=True)
        if self.get_family() is DataSetFamily.heterophilic:
            name = self.name.replace('_', '-').capitalize()
            dataset = [HeterophilousGraphDataset(root=root, name=name, transform=T.ToUndirected())[0]]
        elif self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins]:
            tu_dataset_name = self.name.upper().replace('_', '-')
            root = osp.join(root_dir, 'datasets', tu_dataset_name)
            dataset = torch.load(root + '.pt')
        elif self is DataSet.root_neighbours:
            dataset = [RootNeighboursDataset(seed=seed).get()]
        elif self is DataSet.cycles:
            dataset = CyclesDataset().data
        elif self is DataSet.func:
            dataset_train = LRGBDataset(root=root, name='peptides-func', split='train')
            dataset_val = LRGBDataset(root=root, name='peptides-func', split='val')
            dataset_test = LRGBDataset(root=root, name='peptides-func', split='test')
            dataset_train = apply_transform(dataset=dataset_train, pos_encoder=pos_enc)
            dataset_val = apply_transform(dataset=dataset_val, pos_encoder=pos_enc)
            dataset_test = apply_transform(dataset=dataset_test, pos_encoder=pos_enc)
            dataset = [dataset_train, dataset_val, dataset_test]
            #dataset = PeptidesFunctionalDataset(root=root)
            #print(dataset[0])
            #dataset = apply_transform(dataset=dataset, pos_encoder=pos_enc)
        elif self is DataSet.struct:
            dataset_train = LRGBDataset(root=root, name='peptides-struct', split='train')
            dataset_val = LRGBDataset(root=root, name='peptides-struct', split='val')
            dataset_test = LRGBDataset(root=root, name='peptides-struct', split='test')
            dataset_train = apply_transform(dataset=dataset_train, pos_encoder=pos_enc)
            dataset_val = apply_transform(dataset=dataset_val, pos_encoder=pos_enc)
            dataset_test = apply_transform(dataset=dataset_test, pos_encoder=pos_enc)
            dataset = [dataset_train, dataset_val, dataset_test]
            #dataset = PeptidesStructuralDataset(root=root)
            #dataset = apply_transform(dataset=dataset, pos_encoder=pos_enc)
        elif self is DataSet.pascal:
            dataset_train = LRGBDataset(root=root, name='pascalvoc-sp', split='train')
            dataset_val = LRGBDataset(root=root, name='pascalvoc-sp', split='val')
            dataset_test = LRGBDataset(root=root, name='pascalvoc-sp', split='test')
            # Convert the labels to one hot encoding
            labels = dataset_train.data.y
            num_classes = dataset_train.data.y.max().item() + 1
            dataset_train.data.y = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            labels = dataset_val.data.y
            dataset_val.data.y = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            labels = dataset_test.data.y
            dataset_test.data.y = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            dataset_train = apply_transform(dataset=dataset_train, pos_encoder=pos_enc)
            dataset_val = apply_transform(dataset=dataset_val, pos_encoder=pos_enc)
            dataset_test = apply_transform(dataset=dataset_test, pos_encoder=pos_enc)
            
            
            
            dataset = [dataset_train, dataset_val, dataset_test]
        elif self.get_family() is DataSetFamily.homophilic:
            dataset = [Planetoid(root=root, name=self.name, transform=T.NormalizeFeatures())[0]]
        elif self.get_family() is DataSetFamily.gpp:
            dataset = GPPDataset(root, self.name)
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')
        return dataset

    def select_fold_and_split(self, dataset: List[Data], num_fold: int) -> DatasetBySplit:
        if self.get_family() is DataSetFamily.heterophilic:
            dataset_copy = copy.deepcopy(dataset)
            dataset_copy[0].train_mask = dataset_copy[0].train_mask[:, num_fold]
            dataset_copy[0].val_mask = dataset_copy[0].val_mask[:, num_fold]
            dataset_copy[0].test_mask = dataset_copy[0].test_mask[:, num_fold]
            return DatasetBySplit(train=dataset_copy, val=dataset_copy, test=dataset_copy)
        elif self.get_family() is DataSetFamily.synthetic:
            return DatasetBySplit(train=dataset, val=dataset, test=dataset)
        elif self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins]:
            tu_dataset_name = self.name.upper().replace('_', '-')
            original_fold_dict = json.load(open(f'folds/{tu_dataset_name}_splits.json', "r"))[num_fold]
            model_selection_dict = original_fold_dict['model_selection'][0]
            split_dict = {'train': model_selection_dict['train'], 'val': model_selection_dict['validation'],
                          'test': original_fold_dict['test']}
            dataset_by_splits = [[dataset[idx] for idx in split_dict[split]] for split in DatasetBySplit._fields]
            return DatasetBySplit(*dataset_by_splits)
        elif self is DataSet.func:
            #split_idx = dataset.get_idx_split()
            #dataset_by_splits = [[dataset[idx] for idx in split_idx[split]] for split in DatasetBySplit._fields]
            #return DatasetBySplit(*dataset_by_splits)
            dataset_by_splits = [dataset[0], dataset[1], dataset[2]]
            return DatasetBySplit(*dataset_by_splits)
        elif self is DataSet.struct:
            #split_idx = dataset.get_idx_split()
            #dataset_by_splits = [[dataset[idx] for idx in split_idx[split]] for split in DatasetBySplit._fields]
            dataset_by_splits = [dataset[0], dataset[1], dataset[2]]
            return DatasetBySplit(*dataset_by_splits)
        elif self is DataSet.pascal:
            #split_idx = dataset.get_idx_split()
            #dataset_by_splits = [[dataset[idx] for idx in split_idx[split]]
            #                     for split in DatasetBySplit._fields]
            dataset_by_splits = [dataset[0], dataset[1], dataset[2]]
            return DatasetBySplit(*dataset_by_splits)
            #return DatasetBySplit(train=dataset[0], val=dataset[1], test=dataset[2])
        elif self.get_family() is DataSetFamily.homophilic:
            device = dataset[0].x.device
            with np.load(f'folds/{self.name}_split_0.6_0.2_{num_fold}.npz') as folds_file:
                train_mask = torch.tensor(folds_file['train_mask'], dtype=torch.bool, device=device)
                val_mask = torch.tensor(folds_file['val_mask'], dtype=torch.bool, device=device)
                test_mask = torch.tensor(folds_file['test_mask'], dtype=torch.bool, device=device)

            setattr(dataset[0], 'train_mask', train_mask)
            setattr(dataset[0], 'val_mask', val_mask)
            setattr(dataset[0], 'test_mask', test_mask)

            dataset[0].train_mask[dataset[0].non_valid_samples] = False
            dataset[0].test_mask[dataset[0].non_valid_samples] = False
            dataset[0].val_mask[dataset[0].non_valid_samples] = False
            return DatasetBySplit(train=dataset, val=dataset, test=dataset)
        elif self.get_family() is DataSetFamily.gpp:
            split_idx = dataset.get_idx_split()
            dataset_by_splits = [[dataset[idx] for idx in split_idx[split]] for split in DatasetBySplit._fields]
            return DatasetBySplit(*dataset_by_splits)
        else:
            raise ValueError(f'NotImplemented')

    def get_metric_type(self) -> MetricType:
        if self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins, DataSetFamily.homophilic]\
                or self in [DataSet.roman_empire, DataSet.amazon_ratings, DataSet.cycles]:
            return MetricType.ACCURACY
        elif self in [DataSet.minesweeper, DataSet.tolokers, DataSet.questions]:
            return MetricType.AUC_ROC
        elif self is DataSet.root_neighbours:
            return MetricType.MSE_MAE
        elif self is DataSet.func:
            return MetricType.MULTI_LABEL_AP
        elif self is DataSet.struct:
            return MetricType.MAE
        elif self is DataSet.pascal:
            return MetricType.F1
        elif self is DataSet.diameter:
            return MetricType.logMSEgraph
        elif self in [DataSet.eccentricity, DataSet.sssp]:
            return MetricType.logMSEnode
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def num_after_decimal(self) -> int:
        return 4 if self.get_family() is DataSetFamily.lrgb else 2

    def activation_type(self) -> ActivationType:
        if self.get_family() in [DataSetFamily.heterophilic, DataSetFamily.lrgb]:
            return ActivationType.GELU
        elif self.get_family() is DataSetFamily.gpp:
            return ActivationType.TANH
        else:
            return ActivationType.RELU

    def optimizer(self, model, lr: float, weight_decay: float):
        if self.get_family() in [DataSetFamily.synthetic, DataSetFamily.social_networks,
                                 DataSetFamily.proteins, DataSetFamily.homophilic, DataSetFamily.gpp]:
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.get_family() in [DataSetFamily.lrgb, DataSetFamily.heterophilic]:
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def scheduler(self, optimizer, step_size: Optional[int], gamma: Optional[float], num_warmup_epochs: Optional[int],
                  max_epochs: int):
        if self.get_family() is DataSetFamily.lrgb:
            if self is DataSet.func:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=20, 
                                                                  min_lr=1e-6)
            elif self is DataSet.struct:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=20, 
                                                                  min_lr=1e-6)
            elif self is DataSet.pascal:
                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=20, 
                                                                  min_lr=1e-6)
            else:
                raise ValueError(f'Other lrgb datasets not supported in dataloader')
            
            
            assert num_warmup_epochs is not None, 'cosine_with_warmup_scheduler\'s num_warmup_epochs is None'
            assert max_epochs is not None, 'cosine_with_warmup_scheduler\'s max_epochs is None'
            return cosine_with_warmup_scheduler(optimizer=optimizer, num_warmup_epochs=num_warmup_epochs,
                                                max_epoch=max_epochs)
        elif self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins]:
            assert step_size is not None, 'StepLR\'s step_size is None'
            assert gamma is not None, 'StepLR\'s gamma is None'
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        elif self.get_family() in [DataSetFamily.heterophilic]:
            #return None
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=20, 
                                                                  min_lr=1e-6)
        elif self.get_family() in [DataSetFamily.synthetic, DataSetFamily.homophilic, DataSetFamily.gpp]:
            return None
        else:
            raise ValueError(f'DataSet {self.name} not supported in dataloader')

    def get_split_mask(self, data: Data, batch_size: int, split_mask_name: str) -> Tensor:
        if hasattr(data, split_mask_name):
            return getattr(data, split_mask_name)
        elif self.is_node_based():
            return torch.ones(size=(data.x.shape[0],), dtype=torch.bool)
        else:
            return torch.ones(size=(batch_size,), dtype=torch.bool)

    def get_edge_ratio_node_mask(self, data: Data, split_mask_name: str) -> Tensor:
        if hasattr(data, split_mask_name):
            return getattr(data, split_mask_name)
        else:
            return torch.ones(size=(data.x.shape[0],), dtype=torch.bool)

    def asserts(self, args):
        # model
        pool = (args.model_conf['pool'] if hasattr(args, 'model_conf')  # parallel model selection
                else args.pool) # single experiment
        assert not(self.is_node_based()) or pool is Pool.NONE, "Node based datasets have no pooling"
        assert not(not(self.is_node_based()) and pool is Pool.NONE), "Graph based datasets need pooling"
       
        # dataset dependant parameters
        assert self.get_family() in [DataSetFamily.social_networks, DataSetFamily.proteins] or args.fold is None, \
            'social networks and protein datasets are the only ones to use fold'
        assert self.get_family() not in [DataSetFamily.social_networks, DataSetFamily.proteins] or args.fold is not None, \
            'social networks and protein datasets must specify fold'
        step_size = (args.scheduler_conf['step_size'] if hasattr(args, 'scheduler_conf') # parallel model selection
                     else args.scheduler_step_size) # single experiment
        gamma = (args.scheduler_conf['gamma'] if hasattr(args, 'scheduler_conf') # parallel model selection
                 else args.scheduler_gamma) # single experiment
        num_warmup_epochs = (args.scheduler_conf['num_warmup_epochs'] if hasattr(args, 'scheduler_conf') # parallel model selection
                             else args.scheduler_num_warmup_epochs) # single experiment
        assert self.get_family() is DataSetFamily.proteins or self.get_family() is DataSetFamily.social_networks\
            or (step_size is None and gamma is None),\
                'proteins datasets are the only ones to use scheduler_step_size and scheduler_gamma'
        assert self.get_family() is DataSetFamily.lrgb or (num_warmup_epochs is None),\
            'lrgb datasets are the only ones to use scheduler_num_warmup_epochs'
        assert not(self is DataSet.diameter) or pool is Pool.GPP, "The diameter task needs GPP pooling"
            
        # encoders
        assert self.get_family() is DataSetFamily.lrgb or (args.pos_enc is PosEncoder.NONE), \
            'lrgb datasets are the only ones to use pos_enc'