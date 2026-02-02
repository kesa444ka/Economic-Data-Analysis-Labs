from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


def elbow_method(X, k_range):
    """
    Вычисление значений инерции (суммы квадратов расстояний) для метода локтя
    """
    inertia = []
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, init='random', n_init='auto')
        kmeans_temp.fit(X)
        inertia.append(kmeans_temp.inertia_)

    return inertia


def silhouette_coeff_method(X, k_range):
    """
    Вычисление силуэтных коэффициентов для разного количества кластеров
    """
    results = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        sil_scores = silhouette_samples(X, labels)

        results[k] = {
            "labels": labels,
            "silhouette_values": sil_scores,
            "silhouette_mean": silhouette_score(X, labels)
        }

    return results


def relabel_clusters_by_indicator(df, cluster_col, indicator_col):
    means = df.groupby(cluster_col)[indicator_col].mean().sort_values(ascending=False)
    mapping = {old: new for new, old in enumerate(means.index, 1)}
    return df[cluster_col].map(mapping)
