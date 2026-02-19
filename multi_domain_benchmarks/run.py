import os

root = './'
model = 'BlockSONAR' #'SONAR'
seeds = '30 95 12' # seeds for graph property prediction
patience = 50 # early stopping patience
max_epochs = 200
batch_size = 50
cpus = 5
gpus = 1 # 1 GPU per task2
cuda_id = '0,1,2,3' # GPU IDs for the job

for data in ['questions']: # 'func', 'struct', 'roman_empire, 'amazon_ratings', 'minesweeper', tolokers', 'questions'
    tmp = f'{data}_{model}'
    out = os.path.join(root, f'{tmp}_out')
    err = os.path.join(root, f'{tmp}_err')
    cmd = f'nohup python3 -u main.py --root_dir {root} --dataset {data} --seeds {seeds} '\
          f'--patience {patience} --max_epochs {max_epochs} --batch_size {batch_size} '\
          f'--cpus_per_task {cpus} --gpus_per_task {gpus} '\
          f'>{out} 2>{err} ' # &'
    print(cmd)
    os.system(f'export CUDA_VISIBLE_DEVICES={cuda_id}; {cmd}')
