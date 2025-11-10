from helpers.parse_arguments import parse_arguments
from helpers.dataset_classes.dataset import DataSetFamily
from model_selection import ModelSelection
from helpers.encoders import PosEncoder
from experiments import Experiment
import time, datetime
import os

if __name__ == '__main__':
    t0 = time.time()
    args = parse_arguments()
    print(args)

    assert not (args.debug and args.run_single_experiment), 'Debug mode can be activate only during a parallel model selection'
    gpp_seeds = [41, 95, 12, 35] # https://github.com/gravins/Anti-SymmetricDGN/blob/47ceef92650e67ec01a672f3e07de563ab49ddf4/graph_prop_pred/model_selection.py#L47
    assert set(args.seeds) == set(gpp_seeds) or not(args.dataset.get_family() is DataSetFamily.gpp),\
        f'The experimental seeds for graph property prediction tasks should be {gpp_seeds}'

    # Create all dirs and paths used in the experiments
    # NOTE: The directory containing the data is osp.join(ROOT_DIR, dataset_name, 'datasets')
    args.root_dir = os.path.abspath(args.root_dir) # NOTE: this is required by ray
    args.root_dir = os.path.join(args.root_dir, args.dataset.name)
    args.result_dir = os.path.join(args.root_dir, args.model.name+('_with_pos_enc' if not(args.pos_enc is PosEncoder.NONE) else ''))
    args.ckpt_dir = os.path.join(args.result_dir, 'ckpt')

    os.makedirs(args.root_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    dataset = args.dataset.load(root_dir=args.root_dir, seed=args.data_seed, pos_enc=args.pos_enc) # Download the dataset for the first time
    
    if args.run_single_experiment:
        assert len(args.seeds) == 1, 'Only one experimental seed can be used during a single experiment'
        tmp = vars(args)
        tmp.update({
            'act_type': args.dataset.activation_type(),
            'dataset_encoders': args.dataset.get_dataset_encoders(),
            'dataset_decoders': args.dataset.get_dataset_decoders(),
            'pos_enc': args.pos_enc,
            'in_dim': dataset[0].x.shape[1],
            'out_dim': args.dataset.get_metric_type().get_out_dim(dataset=dataset)
        })
        conf = args.model.get_single_conf(tmp)
        args.model_conf = conf['model_conf']
        args.optimizer_conf = conf['optimizer_conf']
        args.scheduler_conf = conf['scheduler_conf']
        args.model_instance = args.model.get()
        del dataset
        Experiment(args=args).run_()
    else: 
        del dataset
        best_conf = ModelSelection(args=args).run() 
        print('Winning Configuration:', best_conf.iloc[0], sep='\n')

    elapsed = time.time() - t0
    print(f"The experiment required: {datetime.timedelta(seconds=elapsed)}")
