import os
import pandas
from matplotlib import pyplot as plt
import matplotlib 
import numpy as np


plot = {}
plot_std = {}
for data in ['line', 'ring', 'crossed-ring']: 
    tmp, tmp_std = {}, {}
    for distance in [3, 5, 10, 50]:
        try:
            df = pandas.read_csv(f'./results/{data}_{distance}/blocksonar/final_results_blocksonar.csv')
            best_val = df['mean_val_loss'].min()
            index = df[df['mean_val_loss']==best_val].index
            m = df.loc[index, 'mean_test_loss'].values[0]
            m_std = df.loc[index, 'std_test_loss'].values[0]
            tmp[str(distance)] = {'blocksonar': m}
            tmp_std[str(distance)] = {'blocksonar': m_std}
        except:
            tmp[str(distance)] = {'blocksonar': None}
            tmp_std[str(distance)] = {'blocksonar': None}
            
    
    plot[data] = pandas.concat([pandas.read_csv(f'results/baseline_results_{data.replace("-", "")}.csv', index_col='Unnamed: 0'), 
                                pandas.DataFrame(tmp)])
    plot_std[data] = pandas.concat([pandas.read_csv(f'results/baseline_results_std_{data.replace("-", "")}.csv', index_col='Unnamed: 0'), 
                                    pandas.DataFrame(tmp_std)])


avoid = ['sage', 'grama', 'phdgn_conservative']
markers = {
    'gin': 'o',
    'gcn': 's',
    'gat': '^',
    'sage': 'v',
    'swan': 'd',
    'adgn': 'p',
    'phdgn': '*',
    'phdgn_conservative': '*',
    'gps': 'x',
    'grama': '|',
    'blocksonar': '+'
}



mapper={
    'gin': 'gin'.upper(),
    'gcn': 'gcn'.upper(),
    'gat': 'gat'.upper(),
    'sage': 'sage'.upper(),
    'swan': 'swan'.upper(),
    'adgn': 'adgn'.upper(),
    'phdgn': 'phdgn'.upper(),
    'phdgn_conservative': r'PHDGN$_c$',
    'gps': 'gps'.upper(),
    'grama': 'grama'.upper(),
    'blocksonar': 'sonar'.upper()
}

seaborn_colorblind = {
    'gin': '#949494', # grigio
    'gcn': '#de8f05', # arancione
    'sage': '#d55e00', #arancione scuro
    'gat': '#cc78bc', #rosa
    'gps': '#a65628', # marrone
    'swan': '#FFD700', #oro                             '#ece133', # giallino
    'adgn': '#0173b2', #blu
    'phdgn': 'white', # black
    'phdgn_conservative': 'black', # black
    'grama': '#56b4e9',
    'blocksonar': '#029e73', #verde
}
# ['#ca9161', '#fbafe4', '#56b4e9']

# make all plot lines white and set transparent background for saved figures
matplotlib.rcParams['axes.edgecolor'] = 'white'
matplotlib.rcParams['xtick.color'] = 'white'
matplotlib.rcParams['ytick.color'] = 'white'
matplotlib.rcParams['axes.labelcolor'] = 'white'
matplotlib.rcParams['axes.titlecolor'] = 'white'
matplotlib.rcParams['text.color'] = 'white'

# ensure figures/axes are transparent and saved images are transparent by default
matplotlib.rcParams['figure.facecolor'] = 'none'
matplotlib.rcParams['axes.facecolor'] = 'none'
matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['savefig.facecolor'] = 'none'
matplotlib.rcParams['savefig.edgecolor'] = 'none'
matplotlib.rcParams.update({'font.size': 23})

colors = seaborn_colorblind
for log in [True]: 
    fig, ax =plt.subplots(figsize=(20,4.5), ncols=3)
    for i, gtype in enumerate(['line', 'ring', 'crossed-ring']): 
        df = pandas.DataFrame(plot[gtype])
        df_s = pandas.DataFrame(plot_std[gtype])
        x = [int(c) for c in df.columns]
        for k in mapper.keys():
            if k in avoid:
                continue
            if k not in df.index:
                continue
            label = None
            if i == 0:
                label = mapper[k]
            vals = pandas.to_numeric(df.loc[k].values, errors='coerce').astype(float)
            ax[i].plot(x, vals, 
                       marker=markers[k], 
                       color=colors[k], 
                       markersize=9,#7, 
                       linewidth=2.0, 
                       linestyle='dashed' if k == 'phdgn_conservative' else 'solid',
                       label=label)
            # draw colored zone for std if available
            if k in df_s.index:
                yerr = pandas.to_numeric(df_s.loc[k].values, errors='coerce').astype(float)
                if not np.all(np.isnan(yerr)):
                    print(yerr)
                    lower = vals - yerr
                    upper = vals + yerr
                    # when using log scale, fill_between requires strictly positive values
                    safe_lower = lower
                    # mask NaNs before plotting
                    mask = (~np.isnan(safe_lower)) & (~np.isnan(upper)) & (~np.isnan(vals))
                    if mask.any():
                        ax[i].fill_between(np.array(x)[mask], safe_lower[mask], upper[mask],
                                           color=colors[k], alpha=0.18, linewidth=0)

        ax[i].set_xticks(x)
        ax[i].set(
            xlabel='Source-Target distance (#hops)',
            ylabel='Mean Squared Error' if i ==0 else ''
        )
        ax[i].grid('on')
        #ax[i].set_title(f'{gtype}')
        if log:
            ax[i].set_yscale('log')

    lgd=fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
            ncol=8, fancybox=True, shadow=True)

    plt.tight_layout()
    fig.tight_layout()
    if log:
        fig.savefig(f'GraphTransfer_logscale_all.png', bbox_inches='tight')
    else:
        fig.savefig(f'GraphTransfer_all.png', bbox_inches='tight')
    plt.close()
    
