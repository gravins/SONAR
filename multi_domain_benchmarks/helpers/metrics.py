from enum import Enum, auto
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, L1Loss, Module
import torch
from typing import NamedTuple, List
from torchmetrics import Accuracy, AUROC, MeanAbsoluteError, MeanSquaredError, F1Score, AveragePrecision
import math
from torch_geometric.data import Data
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool

import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

class WeightedCrossEntropyLoss(Module):
    """
    Weighted cross-entropy for unbalanced classes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, true, batch=None):
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        true_num = torch.argmax(true, dim=1) if pred.ndim > 1 else true
        label_count = torch.bincount(true_num)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true_num)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            return F.binary_cross_entropy_with_logits(pred, true, weight=weight, reduction='mean')
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(), weight=weight[true])
            return loss

class GenericLoss(Module):
    def __init__(self, loss, node_level=False, apply_log=False, gpp=False):
        super().__init__()
        self.loss = loss
        self.node_level = node_level
        self.apply_log = apply_log
        self.gpp = gpp
        assert not node_level or gpp
 
    def forward(self, pred, target, batch=None):
        if self.gpp:
            target = target.reshape(target.shape[0], 1)
        if self.node_level:
            print(pred[:10])
            # Implementing global add pool of the node losses, reference here
            # https://github.com/cvignac/SMP/blob/62161485150f4544ba1255c4fcd39398fe2ca18d/multi_task_utils/util.py#L99           nodes_in_graph = scatter(torch.ones(batch.shape[0]), batch).unsqueeze(1)
            nodes_in_graph = scatter(torch.ones(batch.shape[0], device=pred.device), batch).unsqueeze(1)
            #nodes_loss = (pred - target) ** 2 # NOTE: this MSE is only used for sssp and eccentricity tasks
            nodes_loss = self.loss(pred, target) # NOTE: this loss is used for all other tasks
            error = global_add_pool(nodes_loss, batch) / nodes_in_graph #average_nodes
            loss = torch.mean(error)
        else:
            loss = self.loss(pred, target)

        return np.log10(loss) if self.apply_log else loss


class LossesAndMetrics(NamedTuple):
    train_loss: float
    val_loss: float
    test_loss: float
    train_metric: float
    val_metric: float
    test_metric: float

    def get_fold_metrics(self):
        return torch.tensor([self.train_metric, self.val_metric, self.test_metric])


class MetricType(Enum):
    """
        an object for the different metrics
    """
    # classification
    ACCURACY = auto()
    MULTI_LABEL_AP = auto()
    AUC_ROC = auto()
    F1 = auto()  # F1 score for multi-label classification

    # regression
    MSE_MAE = auto()
    logMSEnode = auto()
    logMSEgraph = auto()
    MAE = auto()

    def apply_metric(self, scores: np.ndarray, target: np.ndarray, batch=None) -> float: #, batch=None) -> float:
        if isinstance(scores, np.ndarray):
            scores = torch.from_numpy(scores)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        num_classes = scores.size(1)  # target.max().item() + 1
        if self is MetricType.ACCURACY:
            metric = Accuracy(task="multiclass", num_classes=num_classes)
        elif self is MetricType.MULTI_LABEL_AP:
            metric = AveragePrecision(task="multilabel", num_labels=num_classes).to(scores.device)
            result = metric(scores, target.int())
            return result.item()
        elif self is MetricType.F1:
            metric = F1Score(task="multilabel", average='macro', num_labels=num_classes).to(scores.device)
            result = metric(scores, target.int())
            return result.item()
        elif self is MetricType.MAE:
            metric = MeanAbsoluteError()
        elif self is MetricType.MSE_MAE:
            metric = MeanAbsoluteError()
        elif self is MetricType.AUC_ROC:
            metric = AUROC(task="multiclass", num_classes=num_classes)
        elif self is MetricType.logMSEnode:
            metric = GenericLoss(loss=MSELoss(), node_level=True, apply_log=True, gpp=True)
        elif self is MetricType.logMSEgraph:
            metric = GenericLoss(loss=MSELoss(), apply_log=True, gpp=True)
        else:
            raise ValueError(f'MetricType {self.name} not supported')

        metric = metric.to(scores.device)
        if self is MetricType.logMSEnode:
            result = metric(scores, target, batch)
        else:    
            result = metric(scores, target)
        return result.item()

    def is_classification(self) -> bool:
        if self in [MetricType.AUC_ROC, MetricType.ACCURACY, MetricType.MULTI_LABEL_AP, MetricType.F1]:
            return True
        elif self in [MetricType.MSE_MAE, MetricType.logMSEnode, MetricType.logMSEgraph, MetricType.MAE]: #MetricType.logMSE]: #, MetricType.logMSEnode, MetricType.logMSEgraph]:
            return False
        else:
            raise ValueError(f'MetricType {self.name} not supported')

    def is_multilabel(self) -> bool:
        return self is MetricType.MULTI_LABEL_AP or self is MetricType.F1

    def get_task_loss(self):
        if self.is_classification():
            if self.is_multilabel():
                if self is MetricType.F1:
                    return WeightedCrossEntropyLoss()
                else:
                    return GenericLoss(loss=BCEWithLogitsLoss())
            else:
                return GenericLoss(loss=CrossEntropyLoss())
        elif self is MetricType.MSE_MAE:
            return GenericLoss(loss=MSELoss())
        elif self is MetricType.MAE:
            return GenericLoss(loss=L1Loss())
        elif self is MetricType.logMSEnode:
            return GenericLoss(loss=MSELoss(), node_level=True, gpp=True)
        elif self is MetricType.logMSEgraph:
            return GenericLoss(loss=MSELoss(), node_level=False, gpp=True)
        else:
            raise ValueError(f'MetricType {self.name} not supported')

    def get_out_dim(self, dataset: List[Data]) -> int:
        if self.is_classification():
            if self.is_multilabel():
                print(f'get_out_dim: {dataset[0].y.shape[1]}')
                return dataset[0].y.shape[1]
            else:
                return int(max([data.y.max().item() for data in dataset]) + 1)
        elif len(dataset[0].y.shape)==1:
            return 1
        else:
            return dataset[0].y.shape[-1]

    def higher_is_better(self):
        return self.is_classification()

    def src_better_than_other(self, src: float, other: float) -> bool:
        if self.higher_is_better():
            return src > other
        else:
            return src < other

    def get_worst_losses_n_metrics(self) -> LossesAndMetrics:
        if self.is_classification():
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=-math.inf, val_metric=-math.inf, test_metric=-math.inf)
        else:
            return LossesAndMetrics(train_loss=math.inf, val_loss=math.inf, test_loss=math.inf,
                                    train_metric=math.inf, val_metric=math.inf, test_metric=math.inf)
