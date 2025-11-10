from experiments import Experiment
from argparse import Namespace
import pandas as pd
import pickle
import tqdm
import copy
import ray
import os
import gc

class ModelSelection(object):
    def __init__(self, args: Namespace):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)
        
        self.args = args

        self.path_partial_res_pkl = os.path.join(self.result_dir, 'partial_results.pkl')
        self.path_partial_res_csv = os.path.join(self.result_dir, 'partial_results.csv')
        self.path_final_res_csv = os.path.join(self.result_dir, 'model_selection_results.csv')
        
        if not args.debug:
            ray.init(address='local')

        self.df = []
        self.ray_ids = []
        self.metric_type = self.dataset.get_metric_type()
        self.decimal = self.dataset.num_after_decimal()
        dataset = args.dataset.load(root_dir=args.root_dir, seed=args.data_seed, pos_enc=args.pos_enc) # Download for the first time the dataset
        self.conf_list = list(args.model.get_conf(self.dataset, dataset, args.pos_enc))
        del dataset
        num_conf = len(self.conf_list)
        self.pbar = tqdm.tqdm(total=num_conf*len(args.seeds))

    def collect_results(self, metrics_mean, metrics_std, metrics_matrix, exp_args):
        row = {}
        for k in exp_args.keys():
            if isinstance(exp_args[k], dict):
                for sk in exp_args[k]:
                    row[f'{k}_{sk}'] = exp_args[k][sk]
            else:
                row[k] = exp_args[k]
        row.update({
            f'avg_fold_train_{self.metric_type.name}': metrics_mean[0],
            f'avg_fold_val_{self.metric_type.name}': metrics_mean[1],
            f'avg_fold_test_{self.metric_type.name}': metrics_mean[2],
            f'std_fold_train_{self.metric_type.name}': metrics_std[0],
            f'std_fold_val_{self.metric_type.name}': metrics_std[1],
            f'std_fold_test_{self.metric_type.name}': metrics_std[2],
            'single_fold_results': metrics_matrix.tolist()
        })

        self.df.append(row)

        pickle.dump(self.df, open(self.path_partial_res_pkl, 'wb'))
        df_ = pd.DataFrame(self.df).sort_values(f'avg_fold_val_{self.metric_type.name}', ascending=not self.metric_type.is_classification())
        df_.to_csv(self.path_partial_res_csv)

        self.pbar.update(1)
        
        aggregated_df = self.aggregate_res()
        if len(aggregated_df) > 0:
            score = aggregated_df.iloc[0][f'avg_test_{self.metric_type.name}']
            std = aggregated_df.iloc[0][f'std_test_{self.metric_type.name}']
            log_str = f'best conf: {aggregated_df.iloc[0]["conf_id"]}, best avg_test_{self.metric_type.name}: {score}+/-{std}'
            self.pbar.set_description(log_str, refresh=True)


    def aggregate_res(self):
        df_ = pd.DataFrame(self.df)
        # Aggregate results over multiple runs and sort them by best val score
        aggregated_df = []
        for conf_id, gdf in df_.groupby('conf_id'):
            if len(gdf.seed.values) < len(self.seeds):
                continue
            row = {}
            for k in gdf.columns:
                if k == 'single_fold_results' or 'std_fold_' in k:
                    continue
                elif k == 'seed': 
                    row[k] = gdf[k].values 
                elif 'test' in k or 'val' in k or 'train' in k:
                    knew = k.replace('avg_fold_', '') 
                    row[f'avg_{knew}'] = gdf[k].values.mean() if 'confusion_matrix' in k else gdf[k].mean()
                    row[f'std_{knew}'] = gdf[k].values.std() if 'confusion_matrix' in k else gdf[k].std()
                else:
                    row[k] = gdf.iloc[0][k]
            aggregated_df.append(row)

        if len(aggregated_df) > 0:
            aggregated_df = pd.DataFrame(aggregated_df)
            aggregated_df = aggregated_df.sort_values(f'avg_val_{self.metric_type.name}', ascending=not self.metric_type.is_classification())
        return aggregated_df

    def wait_and_collect(self):
        done_id, self.ray_ids = ray.wait(self.ray_ids)
        metrics_mean, metrics_std, metrics_matrix, exp_args = ray.get(done_id[0])
        self.collect_results(metrics_mean, metrics_std, metrics_matrix, exp_args)
        gc.collect()

    def run(self):
        conf_id = 0
        for conf_ in self.conf_list:
            for seed in self.seeds:
                conf = copy.deepcopy(conf_)
                conf.update({
                    'root_dir': self.root_dir,
                    'ckpt_dir': self.ckpt_dir,
                    'conf_id': conf_id,
                    'seed': seed,
                    'data_seed': self.data_seed,
                    'model_conf': conf['model_conf'],
                    'optimizer_conf': conf['optimizer_conf'],
                    'scheduler_conf': conf['scheduler_conf'],
                    'model_instance': self.model.get(),
                    'dataset': self.dataset,
                    'run_single_experiment': False,
                    'patience': self.patience,
                    'max_epochs': self.max_epochs,
                    'batch_size': self.batch_size,
                    'pos_enc': self.pos_enc,
                    'fold': self.fold, # Only for DataSetFamily.social_networks and DataSetFamily.proteins
                    'debug': self.debug
                })
                conf = Namespace(**conf)
                opt = {
                    'num_cpus': self.cpus_per_task, 
                    'num_gpus': self.gpus_per_task 
                }
                exp = Experiment(args=conf)
                if not self.debug:
                    self.ray_ids.append(exp.run.options(**opt).remote(exp))
                else:
                    metrics_mean, metrics_std, metrics_matrix, exp_args = exp.run_()
                    self.collect_results(metrics_mean, metrics_std, metrics_matrix, exp_args)
 
                if self.parallelism is not None and (not self.debug):
                    while len(self.ray_ids) > self.parallelism:
                        self.wait_and_collect()
            conf_id += 1

        while len(self.ray_ids):
            self.wait_and_collect()
              
        ray.shutdown()
        
        final_df = self.aggregate_res()
        final_df.to_csv(self.path_final_res_csv, index=False)
        return final_df