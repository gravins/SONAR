#!/bin/bash


save_dir_GraphProp="./SONAR_NeurIPS25/"
data=GraphProp
gpus=1 # the config requires at least 10% of a free gpu to be scheduled
cpus=5 # the config requires at least 5 free cpus to be scheduled

model=BlockSONAR

export CUDA_VISIBLE_DEVICES=1
task=sssp
nohup python3 -u main.py --cpus $cpus --gpus $gpus --task $task --model_name $model --save_dir $save_dir_GraphProp >$save_dir_GraphProp/out_$model\_$data\_$task  2>$save_dir_GraphProp/err_$model\_$data\_$task &

export CUDA_VISIBLE_DEVICES=2
task=ecc
nohup python3 -u main.py --cpus $cpus --gpus $gpus --task $task --model_name $model --save_dir $save_dir_GraphProp >$save_dir_GraphProp/out_$model\_$data\_$task  2>$save_dir_GraphProp/err_$model\_$data\_$task &

export CUDA_VISIBLE_DEVICES=3
task=diam
nohup python3 -u main.py --cpus $cpus --gpus $gpus --task $task --model_name $model --save_dir $save_dir_GraphProp >$save_dir_GraphProp/out_$model\_$data\_$task  2>$save_dir_GraphProp/err_$model\_$data\_$task &

echo $save_dir_GraphProp/err_$model\_$data\_$task
echo "done"