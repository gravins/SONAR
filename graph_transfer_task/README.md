## How to reproduce the experiments

1) set ```root``` (i.e., root directory that stores the ```data``` and ```results``` folders), the available ```gpus``` ids used for the experiments, and the ```model``` (i.e., the model name defined in ```conf.py```) in ```run_all.py```
2) run: ``` nohup python3 -u run_all.py >out 2> err & ```
3) make plot: ``` python3 plot_graph_transfer_task.py ```