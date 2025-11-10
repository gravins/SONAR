from argparse import ArgumentParser

from helpers.dataset_classes.dataset import DataSet
from helpers.classes import Pool
from helpers.encoders import PosEncoder
import os
from models.model_configs import ModelType

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--root_dir", dest="root_dir", 
                        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        type=str,
                        help='The directory where datasets and checkpoints are stored', 
                        required=False)
    parser.add_argument("--dataset", dest="dataset", default=DataSet.cora, type=DataSet.from_string,
                        choices=list(DataSet), required=False)
    parser.add_argument("--model", dest="model", default=ModelType.BlockSONAR, #SONAR, 
                        type=ModelType.from_string,
                        choices=list(ModelType), required=False)
    parser.add_argument("--data_seed", dest="data_seed", type=int, required=False, default=0)
    parser.add_argument("--pool", dest="pool", default=Pool.MEAN, type=Pool.from_string,
                        choices=list(Pool), required=False)
    
    # parallel model selection
    parser.add_argument("--debug", dest="debug", action='store_true', required=False)
    parser.add_argument("--run_single_experiment", dest="run_single_experiment", action='store_true', required=False,
                        help='if True then no model selection is performed')
    parser.add_argument("--cpus_per_task", dest="cpus_per_task", default=1, type=int, required=False,
                        help='The minimum number of cpus required by for each model config')
    parser.add_argument("--gpus_per_task", dest="gpus_per_task", default=1., type=float, required=False,
                        help='The minimum number of gpus required by for each model config')
    parser.add_argument("--parallelism", dest="parallelism", default=None, type=int, required=False,
                        help='The maximum number of model configs running in parallel. If parallelism is None then no limits are applied')

    # gumbel
    parser.add_argument("--learn_temp", dest="learn_temp", default=False, action='store_true', required=False)
    parser.add_argument("--tau0", dest="tau0", default=0.1, type=float, required=False)
    parser.add_argument("--temp", dest="temp", default=0.01, type=float, required=False)

    # optimization
    parser.add_argument("--patience", dest="patience", default=0, type=int, required=False)
    parser.add_argument("--max_epochs", dest="max_epochs", default=1000, type=int, required=False)
    parser.add_argument("--batch_size", dest="batch_size", default=32, type=int, required=False)
    parser.add_argument("--lr", dest="lr", default=1e-3, type=float, required=False)
    parser.add_argument("--dropout", dest="dropout", default=0.2, type=float, required=False)
    parser.add_argument("--layer_norm", dest="layer_norm", default=False, action='store_true', required=False)

    # env cls parameters
    parser.add_argument("--env_dim", dest="env_dim", default=128, type=int, required=False)
    parser.add_argument("--env_num_layers", dest="env_num_layers", default=2, type=int, required=False)
    parser.add_argument("--env_gamma", dest="env_gamma", default=0.1, type=float, required=False)
    parser.add_argument("--env_epsilon", dest="env_epsilon", default=0.1, type=float, required=False)

    # encoder decoder
    parser.add_argument("--pos_enc", dest="pos_enc", default=PosEncoder.NONE,
                        type=PosEncoder.from_string, choices=list(PosEncoder), required=False)
    parser.add_argument("--dec_num_layers", dest="dec_num_layers", default=1, type=int, required=False)

    # action cls parameters
    parser.add_argument("--num_actions", dest="num_actions", default=4, type=int, required=False)
    parser.add_argument("--act_num_layers", dest="act_num_layers", default=1, type=int, required=False)
    parser.add_argument("--act_gamma", dest="act_gamma", default=0.1, type=float, required=False)
    parser.add_argument("--act_epsilon", dest="act_epsilon", default=0.1, type=float, required=False)

    # reproduce
    parser.add_argument("--seeds", dest="seeds", type=int, nargs='+', default=[0], required=False)
    parser.add_argument('--gpu', dest="gpu", type=int, required=False)

    # dataset dependant parameters 
    parser.add_argument("--fold", dest="fold", default=None, type=int, required=False)

    # optimizer and scheduler
    parser.add_argument("--weight_decay", dest="weight_decay", default=0.1, type=float, required=False)
    ## for steplr scheduler only
    parser.add_argument("--scheduler_step_size", dest="scheduler_step_size", default=None, type=int, required=False)
    parser.add_argument("--scheduler_gamma", dest="scheduler_gamma", default=None, type=float, required=False)
    ## for cosine with warmup scheduler only
    parser.add_argument("--scheduler_num_warmup_epochs", dest="scheduler_num_warmup_epochs", default=None, type=int, required=False)

    return parser.parse_args()
