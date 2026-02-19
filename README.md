# SONAR: Long-Range Graph Propagation Through Information Waves

Official reference implementation of our paper [__"SONAR: Long-Range Graph Propagation Through Information Waves"__](https://openreview.net/pdf?id=Hxfjmc95rl) accepted at NeurIPS 2025.

Please consider citing us

	  @inproceedings{sonar2025,
	    title={{SONAR: Long-Range Graph Propagation Through Information Waves}},
	    author={Alessandro Trenta and Alessio Gravina and Davide Bacciu},
	    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
	    year={2025},
	    url={https://openreview.net/forum?id=Hxfjmc95rl}
	  }

## How to reproduce our results
- Graph Property Prediction Benchmark
  
  ``` cd GraphPropPred; python3 -u main.py --cpus 5 --gpus 1 --task <task_name> --model_name BlockSONAR --save_dir ./SONAR_NeurIPS25/ ``` 

  or

  ```cd GraphPropPrep/; ./run.sh ``` for automated runs

  - Graph Transfer Benchmark
 
    ``` cd graph_transfer_task; python3 -u run_all.py ```

  - LRGB and Heterophilic tasks

  ``` cd multi_domain_benchmarks; python3 run.py ```
