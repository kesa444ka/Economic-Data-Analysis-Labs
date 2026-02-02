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


def elbow_method_plot(k_range, inertia, k=4):
    """
    График метода локтя
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'k={k}')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.legend()
    plt.grid(True)
    plt.show()


def silhouette_plot(X, silhouette_results):
    """
    Cилуэтные диаграммы ("ножи") для разных значений k
    """
    n_plots = len(silhouette_results)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8, 4 * n_rows),
        sharey=False
    )

    axes = np.array(axes).reshape(-1)

    for ax, (k, result) in zip(axes, silhouette_results.items()):
        labels = result["labels"]
        sil_values = result["silhouette_values"]

        y_lower = 0
        yticks = []
        ytick_labels = []
        
        for idx, cluster in enumerate(np.unique(labels), start=1):
            cluster_sil = sil_values[labels == cluster]
            cluster_sil.sort()

            y_upper = y_lower + len(cluster_sil)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_sil,
                alpha=0.7
            )
            
            yticks.append((y_lower + y_upper) / 2)
            ytick_labels.append(idx)
            
            y_lower = y_upper

        ax.axvline(result["silhouette_mean"], linestyle="--", color="red")
        ax.set_title(f"k = {k}\nAvg Silhouette = {silhouette_results[k]["silhouette_mean"]:.3f}")
        ax.set_xlabel("Силуэтный коэффициент")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylabel("Номер кластера")
        
    for ax in axes[n_plots:]:
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()


def silhouette_mean_plot(silhouette_results, k=4):
    """
    График зависимости среднего силуэтного коэффициента от числа кластеров
    """
    ks = list(silhouette_results.keys())
    means = [silhouette_results[k]["silhouette_mean"] for k in ks]

    plt.figure(figsize=(10, 6))
    plt.plot(ks, means, 'bo-', markersize=8)
    plt.axvline(x=k, color='red', linestyle='--', alpha=0.7, label=f'k={k}')
    plt.xlabel("Количество кластеров")
    plt.ylabel("Средний силуэтный коэффициент")
    plt.grid(True)
    plt.show()
