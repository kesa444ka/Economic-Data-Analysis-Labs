import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 


def histogram(data, n_bins):
    data.hist(bins=n_bins, figsize=(20, 20), 
              color='#c99be0',
              edgecolor='black',
              grid=False) 
    plt.suptitle(f'Гистограммы распределения коэффициентов', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()


def box_plot(data):
    n_cols = len(data.columns)
    n_rows = int(np.ceil(n_cols / 3))  # 3 графика в строке
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, column in enumerate(data.columns):
        ax = axes[i]
        data[column].plot.box(ax=ax)
        ax.set_title(f'{column}', fontsize=12)
        # ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Ящики с усами для всех коэффициентов', y = 1.01, fontsize=16)
    plt.tight_layout()
    plt.show()