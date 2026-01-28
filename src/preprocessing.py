import pandas as pd
import numpy as np


def hampel_bounds(data, k=5.2):
    """
    Вычисление границ выбросов по методу Хампеля.
    Значение k=5.2 выбрано по рекомендаций преподавателя
    """
    x_med = np.median(data)
    x_mad = np.median(np.abs(data - x_med))
    
    lower_bound = x_med - k * x_mad
    upper_bound = x_med + k * x_mad
        
    return lower_bound, upper_bound


def hampel_outliers(data, k=5.2):
    """Маска выбросов по методу Хампеля"""
    lower_bound, upper_bound= hampel_bounds(data, k)
    return (data < lower_bound) | (data > upper_bound)


def select_hampel_k(data, k_grid=None, max_share=0.08):
    """
    Множитель k для заданного процента выбросов.
    Значение max_share=0.08 (8%) взято из методических указаний
    """
    if k_grid is None:
        k_grid = np.arange(2.5, 7.0, 0.1)

    n = len(data)
    for k in k_grid:
        share = hampel_outliers(data, k).sum() / n
        if share <= max_share:
            return k
    return k_grid[-1]


def censor_series(data, Kmin, Kmax):
    """
    Цензурирование данных за пределами [Kmin, Kmax].
    Границы задаются экспертно на основе экономической интерпретации
    """
    return data.clip(lower=Kmin, upper=Kmax)


def normalize_series(data, Kmin, Kmax, direction: str = "direct"):
    """Нормировка ряда к диапазону [0, 1] с учетом направления зависимости"""
    if Kmax == Kmin:
        return pd.Series(0.5, index=data.index)

    if direction == "direct":
        return (data - Kmin) / (Kmax - Kmin)
    elif direction == "inverse":
        return (Kmax - data) / (Kmax - Kmin)


def process_koeff(df, column, Kmin, Kmax, direction, max_outlier_share=0.08):
    """Полная обработка одного коэффициента: выявление выбросов, цензурирование, нормировка"""
    x = df[column]

    k = select_hampel_k(x, max_share=max_outlier_share)

    lower, upper = hampel_bounds(x, k)

    x_censored = x.copy()
    x_censored[x < lower] = lower
    x_censored[x > upper] = upper
    x_censored = censor_series(x_censored, Kmin, Kmax)

    x_norm = normalize_series(x_censored, Kmin, Kmax, direction)

    return x_norm
