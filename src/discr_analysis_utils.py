import pandas as pd
import numpy as np


def create_cm_without_test(cm, total_observations, k=4):
    """
    Создание DataFrame из матрицы ошибок (confusion matrix) без тестовой выборки.
    Добавляет столбец с общим количеством наблюдений для каждого класса
    """
    df = pd.DataFrame(cm, 
                      index=[f'Истинный {i}' for i in range(1,k+1)],
                      columns=[f'Предсказанный {i}' for i in range(1,k+1)])
    df['Всего наблюдений'] = total_observations

    return df.round(1)


def create_cm_with_test(cm, total_observations, test_sample, k=4):
    """
    Создание DataFrame из матрицы ошибок с добавлением строки для тестовой выборки.
    Используется для сравнения распределений обучающей и тестовой выборок
    """
    df = pd.DataFrame(
        cm,
        index=[f'Истинный {i}' for i in range(1,k+1)],
        columns=[f'Предсказанный {i}' for i in range(1,k+1)]
    )

    df['Итого'] = total_observations
    df.loc['Тестовая выборка'] = test_sample

    return df.round(1)


def calculate_errors(observs_in_clusters, n, cm):
    """
    Вычисление априорных вероятностей классов и условных вероятностей ошибок классификации.
    Возвращает таблицу с оценками и безусловную вероятность ошибки
    """
    prior_probabilities = observs_in_clusters / n
    conditional_error_rates = 1 - np.diag(cm) / observs_in_clusters  #вер ошибки при усл, что объект принадл классу
    unconditional_error_rate = 1 - np.trace(cm) / n  #общая доля неправильно классифицированных

    error_table = pd.DataFrame({
        'Оценка априорной вероятности класса': prior_probabilities,
        'Оценка условной вероятности ошибки': conditional_error_rates
    })
    error_table.index = error_table.index + 1

    return error_table.round(3), unconditional_error_rate


def deviation_distribution(y_true, y_pred):
    """
    Вычисление распределения отклонений между предсказанными и истинными классами.
    Оценка, на сколько классов отличаются предсказания от истинных значений.
    """
    deviations = y_pred - y_true
    total = len(deviations)
    distribution = {}
    
    for k in range(-3, 4):
        distribution[k] = (np.sum(deviations == k) / total) * 100
    
    distribution['|k|≤1'] = (np.sum(np.abs(deviations) <= 1) / total) * 100
    
    return distribution