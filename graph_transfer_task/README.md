## Graph Transfer Experiment
This task consists on transfering a label from a source node to a target node on different graph topologies. We followed the setup outlined in [_Gravina et al. On Oversquashing in Graph Neural Networks Through The Lens of Dynamical Systems. AAAI 2025_](https://github.com/gravins/SWAN/tree/main/graph_transfer). This code is adapted from the original [SWAN repository](https://github.com/gravins/SWAN/tree/main/graph_transfer).

### How to reproduce the experiments

1) set ```root``` (i.e., root directory that stores the ```data``` and ```results``` folders), the available ```gpus``` ids used for the experiments, and the ```model``` (i.e., the model name defined in ```conf.py```) in ```run_all.py```
2) run: ``` nohup python3 -u run_all.py >out 2> err & ```
3) make plot: ``` python3 plot_graph_transfer_task.py ```
