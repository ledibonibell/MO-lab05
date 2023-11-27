import numpy as np


def bernoulli_criterion(matrix):
    row_with_max_mean = np.argmax(np.mean(matrix, axis=1))
    max_mean = np.max(np.mean(matrix, axis=1))
    return row_with_max_mean + 1, max_mean


def wald_criterion(matrix):
    min_values = np.min(matrix, axis=1)
    max_min_value = np.max(min_values)
    row_with_max_min_value = np.argmax(min_values)
    return row_with_max_min_value + 1, max_min_value


def max_criterion(matrix):
    max_values = np.max(matrix, axis=1)
    max_max_value = np.max(max_values)
    row_with_max_max_value = np.argmax(max_values)
    return row_with_max_max_value + 1, max_max_value


def hurwicz_criterion(matrix, a):
    min_values = np.min(matrix, axis=1)
    max_values = np.max(matrix, axis=1)
    row_with_best_value = np.argmax(a * min_values + (1 - a) * max_values)
    hurwicz = np.max(a * min_values + (1 - a) * max_values)
    return row_with_best_value + 1, hurwicz


def savage_criterion(matrix):
    max_values = np.max(matrix, axis=1)
    min_max_value = np.min(max_values)
    row_with_min_max_value = np.argmin(max_values)
    return row_with_min_max_value + 1, max_values, min_max_value


'''########
Подсчет значений
########'''

game_matrix = np.array([[8, 12, 4, 17],
                        [1, 6, 19, 19],
                        [17, 11, 11, 6],
                        [8, 10, 15, 17],
                        [1, 16, 2, 16]])

risk_table = np.max(game_matrix, axis=0) - game_matrix
alpha = 0.5

'''########
Вывод результатов
########'''

print("Начальная матрица стратегий:", "\n", game_matrix)

print(f"\nМетод Бернулли (принцип недостаточного основания): \nНомер стратегии игрока - {bernoulli_criterion(game_matrix)[0]} \nМатематическое ожидание - {bernoulli_criterion(game_matrix)[1]}")

print(f"\nМетод Вальда (пессимистический): \nНомер стратегии игрока - {wald_criterion(game_matrix)[0]} \nНижняя цена игры - {wald_criterion(game_matrix)[1]}")

print(f"\nМетод максимума (оптимистический): \nНомер стратегии игрока - {max_criterion(game_matrix)[0]} \nМаксимальный выигрыш - {max_criterion(game_matrix)[1]}")

print(f"\nМетод Гурвица: \nНомер стратегии игрока - {hurwicz_criterion(game_matrix, alpha)[0]} \nОптимальное значение - {hurwicz_criterion(game_matrix, alpha)[1]}")

print(f"\nМетод Севиджа (критерий риска): \n{risk_table} \n\nНомер стратегии игрока - {savage_criterion(risk_table)[0]} \nТаблица максимумов - {savage_criterion(risk_table)[1]} \nОптимальное значение - {savage_criterion(risk_table)[2]}")
