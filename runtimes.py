import torch
import torch.nn.functional as F

from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.nn import GCNConv, GPSConv
from GraphPropPred.models.sonar import SONARConv
import time
import numpy
import pickle, pandas
import gc

class GNN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, conv_name, nlayers, conv_params={}):
        super().__init__()
        self.emb = torch.nn.Linear(input_dim, hidden_dim)
        self.convs = []
        self.mlps = []
        for _ in range(nlayers):
            if 'sonar' in conv_name:
                self.convs.append(SONARConv(**conv_params))
                self.mlps.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(hidden_dim, hidden_dim),
                        getattr(torch.nn, conv_params['activ_fun'])(),
                        torch.nn.Linear(hidden_dim, hidden_dim)
                    )
                )
            elif conv_name == 'gcn':
                self.convs.append(GCNConv(**conv_params))
            elif conv_name == 'gps':
                nn = GCNConv(hidden_dim, hidden_dim)
                self.convs.append(GPSConv(hidden_dim, nn, heads=2,#heads=4 if hidden_dim % 4 == 0 else 3 if hidden_dim%3 == 0 else 1,
                                          attn_type='multihead', attn_kwargs={}, # pyg >= 2.4
                                          #attn_dropout=0.0, # pyg < 2.4.0
                                          norm='layer'))
            else:
                raise ValueError(conv_name)
        self.convs = torch.nn.ModuleList(self.convs)
        self.mlps = torch.nn.ModuleList(self.mlps) if len(self.mlps) > 0 else None
        self.dec = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.emb(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.mlps is not None:
                x = self.mlps[i](x)
        x = self.dec(x)
        return x


#############################################

device = torch.device('cuda')
dataset = HeterophilousGraphDataset('./data_runtimes', "Roman-empire")
data = dataset[0]
in_dim = 300 # 300
out_dim = 18 # 18
fold_id = 0
mask_train = data.train_mask[:,fold_id]
mask_valid = data.val_mask[:,fold_id]
mask_test = data.test_mask[:,fold_id]



def train(data, mask_train):
    model.train()
    optimizer.zero_grad()
    pred = model(data)
    F.nll_loss(pred[mask_train], data.y[mask_train]).backward()
    optimizer.step()


@torch.no_grad()
def test(data, mask_test):
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[mask_test] == data.y[mask_test]).sum()
    acc = int(correct) / int(mask_test.sum())
    return acc

csv = {}
import os
if not os.path.exists('runtimes.pkl'):
    for num_layers in [2, 4, 8, 16, 32]:
        csv[num_layers] = {}
        for conv_name in ['gcn', 'sonar', 'gps', 'sonar2']:
            hidden_dim = 10
            num_params = 0
            while num_params < 100000:
                hidden_dim += 1
                if conv_name== 'gcn':
                    hidden_dim_use = hidden_dim
                    params = {
                        'in_channels': hidden_dim_use,
                        'out_channels': hidden_dim_use,
                    }
                    nl = num_layers
                elif 'sonar' in conv_name:
                    hidden_dim_use = hidden_dim
                    params = {
                        'in_channels': hidden_dim_use,
                        'edge_channels': 0,
                        'num_iters': num_layers,
                        'epsilon': 0.1,
                        'activ_fun': 'Identity',
                        'normalization': None,
                        'use_dissipation': True,
                        'use_forcing': True,
                        'fix_resistance': False,
                        'bias': True
                    }
                    nl = 2 if conv_name == 'sonar2' else 1
                else:
                    hidden_dim_use = hidden_dim - hidden_dim % 2
                    nl = num_layers
                    params = {}

                model = GNN(input_dim=in_dim, 
                            output_dim=out_dim, 
                            hidden_dim=hidden_dim_use, 
                            conv_name=conv_name, 
                            nlayers=nl, 
                            conv_params=params)
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            torch.cuda.empty_cache()
            gc.collect()
            
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)
            data.to(device)

            train_time = []
            test_time = []
            best_acc = 0
            try:
                for epoch in range(11):
                    torch.cuda.reset_peak_memory_stats(device)
                    t = time.time()
                    train(data, mask_train)
                    train_time.append(time.time() - t)
                    # Record GPU memory usage for training
                    gpu_memory_used_train = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
                    torch.cuda.reset_peak_memory_stats(device)
                    print(f"Epoch {epoch}: GPU Memory Used during Training: {gpu_memory_used_train:.2f} MB")
                    t = time.time()
                    test_acc = test(data, mask_test)
                    test_time.append(time.time() - t)
                    # Record GPU memory usage for testing
                    gpu_memory_used_test = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
                    torch.cuda.reset_peak_memory_stats(device)
                    print(f"Epoch {epoch}: GPU Memory Used during Testing: {gpu_memory_used_test:.2f} MB")
                    if test_acc > best_acc:
                        best_acc = test_acc

                csv[num_layers][conv_name] = {
                    'nparams': num_params,
                    'hidden_dim': hidden_dim,
                    'train_time avg': numpy.average(train_time[1:]),
                    'train_time std': numpy.std(train_time[1:]),
                    'test_time avg': numpy.average(test_time[1:]),
                    'test_time std': numpy.std(test_time[1:]),
                    'train_memory avg': numpy.average([gpu_memory_used_train for _ in range(1, len(train_time))]),
                    'train_memory std': numpy.std([gpu_memory_used_train for _ in range(1, len(train_time))]),
                    'test_memory avg': numpy.average([gpu_memory_used_test for _ in range(1, len(test_time))]),
                    'test_memory std': numpy.std([gpu_memory_used_test for _ in range(1, len(test_time))])
                }
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Out of Memory for {conv_name} with {num_layers} layers and hidden_dim {hidden_dim}")
                    csv[num_layers][conv_name] = {
                        'nparams': num_params,
                        'hidden_dim': hidden_dim,
                        'train_time avg': 'OOM',
                        'train_time std': 'OOM',
                        'test_time avg': 'OOM',
                        'test_time std': 'OOM',
                        'train_memory avg': 'OOM',
                        'train_memory std': 'OOM',
                        'test_memory avg': 'OOM',
                        'test_memory std': 'OOM'
                    }
                    
                else:
                    raise e
        pandas.DataFrame(csv[num_layers]).to_csv(f'{num_layers}.csv', index=True)   
        torch.cuda.empty_cache()
        gc.collect()
    pickle.dump(csv, open('runtimes.pkl', 'wb'))

else:
    csv = pickle.load(open('runtimes.pkl', 'rb'))
import matplotlib.pyplot as plt

# Extract data for plotting
models = ['gcn', 'sonar', 'gps', 'sonar2']
num_layers_list = [4, 8, 16, 32]

train_time_avg = {model: [csv[nl][model]['train_time avg'] for nl in num_layers_list if csv[nl][model]['train_time avg'] != 'OOM'] for model in models}
train_time_std = {model: [csv[nl][model]['train_time std'] for nl in num_layers_list if csv[nl][model]['train_time std'] != 'OOM'] for model in models}
test_time_avg = {model: [csv[nl][model]['test_time avg'] for nl in num_layers_list if csv[nl][model]['test_time avg'] != 'OOM'] for model in models}
test_time_std = {model: [csv[nl][model]['test_time std'] for nl in num_layers_list if csv[nl][model]['test_time std'] != 'OOM'] for model in models}

# Plot train_time avg with std
plt.figure(figsize=(10, 6))
for model in models:
    plt.errorbar(num_layers_list[:len(train_time_avg[model])], train_time_avg[model], yerr=train_time_std[model], label=model, marker='o', capsize=5)
plt.xlabel('Number of Layers')
plt.ylabel('Train Time (avg)')
plt.title('Train Time vs Number of Layers')
plt.legend()
plt.grid(True)
plt.savefig('train_time_vs_layers.png')
plt.show()

# Plot test_time avg with std
plt.figure(figsize=(10, 6))
for model in models:
    plt.errorbar(num_layers_list[:len(test_time_avg[model])], test_time_avg[model], yerr=test_time_std[model], label=model, marker='o', capsize=5)
plt.xlabel('Number of Layers')
plt.ylabel('Test Time (avg)')
plt.title('Test Time vs Number of Layers')
plt.legend()
plt.grid(True)
plt.savefig('test_time_vs_layers.png')
plt.show()

