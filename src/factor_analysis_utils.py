import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer


def create_factor_summary_df(eigenvalues, explained_variance, cumulative_variance):
    """
    Создание сводной таблицы с основными статистиками факторного анализа
    """
    stats_df = pd.DataFrame({
        'Собственное значение': eigenvalues,
        '% объясненной дисперсии': explained_variance,
        'Кумулятивный % объясненной дисперсии': cumulative_variance
    })
    stats_df.index = stats_df.reset_index().index + 1
    return stats_df


def filter_loadings(df, threshold=0.4):
    """
    Фильтрация факторных нагрузок по порогу значимости
    """
    filtered_df = df.copy()
    
    for col in filtered_df.columns:
        for idx in filtered_df.index:
            value = filtered_df.loc[idx, col]
            if abs(value) < threshold:
                filtered_df.loc[idx, col] = '' 
            else:
                filtered_df.loc[idx, col] = f"{value:.3f}"

    return filtered_df


def run_pca_with_kaiser(X):
    """
    Выполнение PCA и определение числа факторов по критерию Кайзера
    """
    pca = PCA()
    pca.fit(X)

    eigenvalues = pca.explained_variance_
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    n_factors = (eigenvalues > 1).sum()

    return pca, n_factors, eigenvalues, explained_var, cumulative_var


def run_rotated_factor_analysis(X, n_factors, rotation='quartimax'):
    """
    Факторный анализ с вращением
    """
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation,
        method='principal'
    )
    fa.fit(X)

    loadings = fa.loadings_
    eigenvalues = (loadings ** 2).sum(axis=0)

    explained_variance = eigenvalues / X.shape[1] * 100
    cumulative_variance = explained_variance.cumsum()

    return fa, loadings, eigenvalues, explained_variance, cumulative_variance


def calculate_integral_indicator(factor_scores, explained_variance):
    """
    Вычисление интегрального показателя на основе взвешенной суммы факторных оценок.
    Веса определяются долей объясненной дисперсии каждого фактора
    """
    weights = explained_variance / 100
    I = (factor_scores * weights).sum(axis=1)
    return I
