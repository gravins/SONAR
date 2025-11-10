from enum import Enum, auto
from models.sonar import BlockSONAR #SONAR
from helpers.dataset_classes.dataset import DataSetFamily, DataSet
from helpers.encoders import GPPDecoders
from helpers.utils import cartesian_product
from helpers.classes import EnvArgs
from helpers.classes import Pool


class ModelType(Enum):

    BlockSONAR = auto()

    @staticmethod
    def from_string(s: str):
        try:
            return ModelType[s]
        except KeyError:
            raise ValueError()
        
    def get(self):
        if self is ModelType.BlockSONAR:
            return BlockSONAR
        else:
            raise ValueError(f'ModelType {self.name} not supported')
        
    def get_single_conf(self, params):
        #if self is ModelType.SONAR:
        if self is ModelType.BlockSONAR:
            if 'hid_dim' not in params.keys():
                
                import numpy as np
                base_params = 4*params['num_layers']
                if params['use_dissipation']:
                    base_params += params['num_layers']
                if params['use_forcing']:
                    base_params += params['num_layers']
                    
                if not params['aggregate_edge_features']:
                    base_params += 1
                params['hid_dim'] = np.sqrt((5e5) / base_params).astype(int)
                if params['aggregate_edge_features']:
                    params['hid_dim'] = params['hid_dim'] // 3
                
            return {
                    'model_conf':{
                        'env_args': EnvArgs(in_dim=params['in_dim'], hid_dim=params['hid_dim'], out_dim=params['out_dim'], #edge_dim=params['edge_dim'],
                                            num_layers=params['num_layers'],  #NOTE: this is used as number of blocks
                                            num_iters=params['num_iters'], 
                                            epsilon=params['epsilon'], layer_norm=params['layer_norm'],  
                                            dropout=params['dropout'],  act_type=params['act_type'], pos_enc=params['pos_enc'],  
                                            dataset_encoders=params['dataset_encoders'], dataset_decoders=params['dataset_decoders'], 
                                            dec_num_layers=params['dec_num_layers'], 
                                            normalization=params['normalization'],
                                            fix_resistance=params['fix_resistance'],
                                            use_dissipation=params['use_dissipation'],
                                            use_forcing=params['use_forcing'],
                                            residual=params['residual'],
                                            pre_act_norm=params['pre_act_norm'],
                                            aggregate_edge_features=params['aggregate_edge_features']),
                        'pool': params['pool']
                    },
                    'optimizer_conf': {
                        'lr': params['lr'],
                        'weight_decay': params['weight_decay']
                    },
                    'scheduler_conf': {
                        'step_size': params['scheduler_step_size'], 
                        'gamma': params['scheduler_gamma'],
                        'num_warmup_epochs': params['scheduler_num_warmup_epochs']
                    }
                }
        else:
            raise ValueError(f'ModelType {self.name} not supported')
        
    def get_conf(self, dataset_type, dataset, pos_enc):
        fixed_hyperparams = {
            'act_type': dataset_type.activation_type(),
            'dataset_encoders': dataset_type.get_dataset_encoders(),
            'dataset_decoders': dataset_type.get_dataset_decoders(),
            'pos_enc': pos_enc,
            'in_dim': dataset[0].x.shape[1],
            'out_dim': dataset_type.get_metric_type().get_out_dim(dataset=dataset),
            #'edge_dim': edge_dim # NOTE: DataSetEncoders processes by default edge_weights if they exists (i.e., in Peptide Func) and return a new emb of dim hid_dim
        }
    
        if dataset_type.get_family() is DataSetFamily.gpp:
            shared_hyperparams = { # NOTE: these hyperparams are taken from https://github.com/gravins/Anti-SymmetricDGN/tree/main/graph_prop_pred
                'lr': [0.003], 
                'weight_decay': [1e-6],
                'scheduler_step_size': [None], 
                'scheduler_gamma': [None],
                'scheduler_num_warmup_epochs': [None], 
                'hid_dim': [30, 20, 10],
                'num_layers': [20, 10, 5], 
                'dropout': [0.],
                'layer_norm': [False], 
                'dec_num_layers': [1] if fixed_hyperparams['dataset_decoders'] is GPPDecoders.NONE else [None],
                'pool': [Pool.GPP] if dataset_type is DataSet.diameter else [Pool.NONE]
            }

            #if self is ModelType.SONAR:
            if self is ModelType.BlockSONAR:
                hyperparams = {
                    'epsilon': [1., 1e-1, 1e-3],
                    'normalization': [None, 'sym', 'rw'],
                    'use_dissipation': [True, False],
                    'use_forcing': [True, False],
                    'fix_resistance': [True, False]
                }
                hyperparams.update(shared_hyperparams)
                hyperparams['num_iters'] = hyperparams['num_layers']
                hyperparams['num_layers'] = [1, 2] # NOTE: this is used for the number of blocks
            else:
                raise ValueError(f'ModelType {self.name} not supported')

            for params in cartesian_product(hyperparams):
                params.update(fixed_hyperparams)
                yield self.get_single_conf(params)
        elif dataset_type.get_family() is DataSetFamily.lrgb:
            shared_hyperparams = {
                'lr': [0.001],
                # For LRGB the hid_dim is calculated above to fit approx 500k parameters
                'weight_decay': [0.],
                'scheduler_step_size': [None], 
                'scheduler_gamma': [None],
                'scheduler_num_warmup_epochs': [5], 
                'dropout': [0.2],
                'residual': [True, False],
                'num_iters': [8,12,16],
                'num_layers': [3,4],
                'pre_act_norm': [True, False],
                'layer_norm': [True,False],
                'epsilon': [0.005,0.01],
                'aggregate_edge_features': [True],
                'dec_num_layers': [1] if fixed_hyperparams['dataset_decoders'] is GPPDecoders.NONE else [None],
                'pool': [Pool.MEAN] if dataset_type in [DataSet.func, DataSet.struct] else [Pool.NONE] # OLD uses SUM
            }
            
            if self is ModelType.BlockSONAR:
                hyperparams = {
                    'normalization': [None], #TODO usare none se non va bene nulla
                    'use_dissipation': [True, False],
                    'use_forcing': [True, False],
                    'fix_resistance': [True, False]
                }
                hyperparams.update(shared_hyperparams)
                
            for params in cartesian_product(hyperparams):
                params.update(fixed_hyperparams)
                yield self.get_single_conf(params)
                
        elif dataset_type.get_family() is DataSetFamily.heterophilic:
            shared_hyperparams = { 
                'lr': [0.0005], 
                'weight_decay': [0.],
                'hid_dim': [64,128,256,512],
                'scheduler_step_size': [None], 
                'scheduler_gamma': [None],
                'scheduler_num_warmup_epochs': [None], 
                'pre_act_norm': [False, True],
                'aggregate_edge_features': [False], # Heterophilic datasets do not have edge features, if set to true model will not work
                'num_layers': [7,6,5],
                'num_iters': [2,1],
                'dropout': [0.5],
                'epsilon': [0.1, 0.05],
                'layer_norm': [True, False],
                'residual': [False, True],
                'dec_num_layers': [5] if fixed_hyperparams['dataset_decoders'] is GPPDecoders.NONE else [None],
                'pool': [Pool.MEAN] if dataset_type in [DataSet.func, DataSet.struct] else [Pool.NONE] # OLD uses SUM
            }
            import numpy as np
            
            if self is ModelType.BlockSONAR:
                hyperparams = {
                    'normalization': [None], #TODO usare none se non va bene nulla
                    'use_dissipation': [True, False],
                    'use_forcing': [True, False],
                    'fix_resistance': [False, True] # Poi provare con false e mettere altro
                }
                hyperparams.update(shared_hyperparams)
                
            for params in cartesian_product(hyperparams):
                params.update(fixed_hyperparams)
                yield self.get_single_conf(params)
            
        else: 
            raise NotImplementedError()