from argparse import Namespace
import torch
import sys
import tqdm
from typing import Tuple, Any
from torch_geometric.loader import DataLoader
from torch import Tensor
from torch_geometric.typing import OptTensor
import numpy as np
import os

#from helpers.classes import GumbelArgs, EnvArgs, ActionArgs, ActivationType
from helpers.metrics import LossesAndMetrics
from helpers.utils import set_seed, optimizer_to
from helpers.dataset_classes.dataset import DatasetBySplit, DataSet
import ray


class Experiment(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)
            if args.run_single_experiment or args.debug: print(f"{arg}: {value_arg}")

        # parameters
        self.metric_type = self.dataset.get_metric_type()
        self.decimal = self.dataset.num_after_decimal()
        self.task_loss = self.metric_type.get_task_loss()

        # asserts
        self.dataset.asserts(args)

    @ray.remote(num_cpus=1)
    def run(self):
        return self.run_()
    
    def run_(self) -> Tensor:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        set_seed(seed=self.seed)

        dataset = self.dataset.load(root_dir=self.root_dir, seed=self.data_seed, pos_enc=self.pos_enc)
        
        #if self.metric_type.is_multilabel():
        #    dataset.data.y = dataset.data.y.to(dtype=torch.float)

        folds = self.dataset.get_folds(fold=self.fold)

        # locally used parameters
        self.out_dim = self.metric_type.get_out_dim(dataset=dataset)

        self.exp_args = {
            'conf_id': self.conf_id,
            'seed': self.seed,
            'model_conf': self.model_conf,
            'optimizer_conf': self.optimizer_conf,
            'scheduler_conf': self.scheduler_conf
        }
        
        # folds
        metrics_list = []
        for num_fold in folds:
            self.ckpt_path = os.path.join(self.ckpt_dir, f'conf_{self.conf_id}_seed_{self.seed}_fold_{num_fold}.pth') 

            dataset_by_split = self.dataset.select_fold_and_split(num_fold=num_fold, dataset=dataset)
            best_losses_n_metrics = self.single_fold(dataset_by_split=dataset_by_split, #gumbel_args=gumbel_args,
                                                     #env_args=env_args, action_args=action_args, 
                                                     num_fold=num_fold)

            ## print final
            if self.run_single_experiment:
                print_str = f'Fold {num_fold}/{len(folds)}'
                for name in best_losses_n_metrics._fields:
                    print_str += f",{name}={round(getattr(best_losses_n_metrics, name), self.decimal)}"
                print(print_str)
                print()
            metrics_list.append(best_losses_n_metrics.get_fold_metrics())

        metrics_matrix = torch.stack(metrics_list, dim=0)  # (F, 3)
        metrics_mean = torch.mean(metrics_matrix, dim=0).tolist()  # (3,)
        metrics_std = torch.std(metrics_matrix, dim=0).tolist() if len(folds) > 1 else [0., 0., 0.] # (3,)

        if self.run_single_experiment:
            # prints
            print(f'Final Rewired train={round(metrics_mean[0], self.decimal)},'
                f'val={round(metrics_mean[1], self.decimal)},'
                f'test={round(metrics_mean[2], self.decimal)}')
            print(f'Final Rewired train={round(metrics_mean[0], self.decimal)}+-{round(metrics_std[0], self.decimal)},'
                    f'val={round(metrics_mean[1], self.decimal)}+-{round(metrics_std[1], self.decimal)},'
                    f'test={round(metrics_mean[2], self.decimal)}+-{round(metrics_std[2], self.decimal)}')
            return metrics_mean

        return metrics_mean, metrics_std, metrics_matrix, self.exp_args

    def single_fold(self, dataset_by_split: DatasetBySplit, #gumbel_args: GumbelArgs, env_args: EnvArgs,
                    #action_args: ActionArgs, 
                    num_fold: int) -> LossesAndMetrics:
        model = self.model_instance(**self.model_conf).to(device=self.device)

        optimizer = self.dataset.optimizer(model=model, lr=self.optimizer_conf['lr'], weight_decay=self.optimizer_conf['weight_decay'])
        scheduler = self.dataset.scheduler(optimizer=optimizer, 
                                           step_size=self.scheduler_conf['step_size'], 
                                           gamma=self.scheduler_conf['gamma'],
                                           num_warmup_epochs=self.scheduler_conf['num_warmup_epochs'], 
                                           max_epochs=self.max_epochs)

        pbar = tqdm.tqdm(total=self.max_epochs, file=sys.stdout) if self.run_single_experiment or self.debug else None
        best_losses_n_metrics = self.train_and_test(dataset_by_split=dataset_by_split, model=model,
                                                    optimizer=optimizer, scheduler=scheduler, 
                                                    pbar=pbar, num_fold=num_fold)
        return best_losses_n_metrics

    def train_and_test(self, dataset_by_split: DatasetBySplit, model, optimizer, scheduler, pbar, num_fold: int) -> LossesAndMetrics:
        train_loader = DataLoader(dataset_by_split.train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset_by_split.val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset_by_split.test, batch_size=self.batch_size, shuffle=True)

        best_losses_n_metrics = self.metric_type.get_worst_losses_n_metrics()
        best_epoch = 0
        history = []

        # LOAD previuos ckpt if exists
        if os.path.exists(self.ckpt_path):
            # Load the existing checkpoint
            print(f'Loading {self.ckpt_path}')
            ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            best_epoch = ckpt['epoch']
            best_losses_n_metrics = ckpt['best_losses_n_metrics']
            history = ckpt['history']

            if 'train_ended' in ckpt and ckpt['train_ended']:
                print(f'Training has ended for {self.ckpt_path}. I am not overriding it.')
                return best_losses_n_metrics

            model.load_state_dict(ckpt['model_state_dict'])
            if scheduler is not None: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            optimizer_to(optimizer, self.device) # Map the optimizer to the current device
            if self.run_single_experiment or self.debug: pbar.update(n=best_epoch)

        model.to(self.device)

        for epoch in range(best_epoch, self.max_epochs):
            self.train(train_loader=train_loader, model=model, optimizer=optimizer)
            train_loss, train_metric = self.test(loader=train_loader, model=model, split_mask_name='train_mask')
            if self.dataset.is_expressivity():
                val_loss, val_metric = train_loss, train_metric
                test_loss, test_metric = train_loss, train_metric
            else:
                val_loss, val_metric = self.test(loader=val_loader, model=model, split_mask_name='val_mask')
                test_loss, test_metric = self.test(loader=test_loader, model=model, split_mask_name='test_mask')

            losses_n_metrics = \
                LossesAndMetrics(train_loss=train_loss, val_loss=val_loss, test_loss=test_loss,
                                 train_metric=train_metric, val_metric=val_metric, test_metric=test_metric)
            if scheduler is not None:
                scheduler.step(losses_n_metrics.val_metric)

            history.append(losses_n_metrics)
            #print(f'Current scheduler lr: {scheduler.get_last_lr()}')
            #print(f'Epoch {epoch} - train_metric: {train_metric}, val_metric: {val_metric}, test_metric: {test_metric}')

            # best metrics
            if epoch == 0 or self.metric_type.src_better_than_other(src=losses_n_metrics.val_metric,
                                                      other=best_losses_n_metrics.val_metric):
                # the metric has improved
                best_losses_n_metrics = losses_n_metrics
                best_epoch = epoch
                torch.save({
                    'train_ended': False,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'best_losses_n_metrics': best_losses_n_metrics,
                    'history': history,
                    'exp_args': self.exp_args
                }, self.ckpt_path)
                print(f'Epoch {epoch} - New best val_metric: {best_losses_n_metrics.val_metric}')
            print(f'Epoch {epoch} - train_metric: {train_metric}, val_metric: {val_metric}, test_metric: {test_metric}')

            if self.patience and epoch - best_epoch > self.patience:
                break

            if self.run_single_experiment or self.debug:
                # prints
                log_str = f'Split: {num_fold}, epoch: {epoch}'
                for name in losses_n_metrics._fields:
                   log_str += f",{name}={round(getattr(losses_n_metrics, name), self.decimal)}"
                log_str += f"({round(best_losses_n_metrics.test_metric, self.decimal)})"
                pbar.set_description(log_str)
                pbar.update(n=1)

        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        ckpt['train_ended'] = True
        torch.save(ckpt, self.ckpt_path)

        return best_losses_n_metrics

    def train(self, train_loader, model, optimizer):
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            node_mask = self.dataset.get_split_mask(data=data, batch_size=data.num_graphs,
                                                    split_mask_name='train_mask').to(self.device)
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores = model(data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device),
                           batch=data.batch.to(device=self.device), edge_attr=edge_attr,
                           pestat=self.pos_enc.get_pe(data=data, device=self.device))

            train_loss = self.task_loss(scores[node_mask], data.y.to(device=self.device)[node_mask],
                                        data.batch.to(device=self.device) if hasattr(data, 'batch') else None)

            # backward
            train_loss.backward()
            if self.dataset.clip_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

    def test(self, loader, model, split_mask_name: str) -> Tuple[float, Any]:
        model.eval()

        total_loss, total_metric = 0, 0
        total_scores = np.empty(shape=(0, self.out_dim))
        total_y = None
        total_batch = None
        for i, data in enumerate(loader):
            node_mask = self.dataset.get_split_mask(data=data, batch_size=data.num_graphs,
                                                    split_mask_name=split_mask_name).to(device=self.device)
            edge_attr = data.edge_attr
            if data.edge_attr is not None:
                edge_attr = edge_attr.to(device=self.device)

            # forward
            scores = model(data.x.to(device=self.device), edge_index=data.edge_index.to(device=self.device),
                           edge_attr=edge_attr, batch=data.batch.to(device=self.device),
                           pestat=self.pos_enc.get_pe(data=data, device=self.device))

            eval_loss = self.task_loss(scores, data.y.to(device=self.device), 
                                       data.batch.to(device=self.device) if hasattr(data, 'batch') else None)

            # analytics
            if total_batch is None:
                total_batch = (data.batch.detach().cpu().numpy() if hasattr(data, 'batch') 
                               else np.ones((1), dtype=int)) # NOTE: If hasattr(data, 'batch') == False, then total_batch is not used
            else: 
                total_batch = np.concatenate((total_batch, data.batch.detach().cpu().numpy()+(i*self.batch_size))) 
            total_scores = np.concatenate((total_scores, scores[node_mask].detach().cpu().numpy()))
            if total_y is None:
                total_y = data.y.to(device=self.device)[node_mask].detach().cpu().numpy()
            else:
                total_y = np.concatenate((total_y, data.y.to(device=self.device)[node_mask].detach().cpu().numpy()))

            total_loss += eval_loss.item() * data.num_graphs

        metric = self.metric_type.apply_metric(scores=total_scores, target=total_y, batch=total_batch)

        loss = total_loss / len(loader.dataset)
        return loss, metric
