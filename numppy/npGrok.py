import numpy as np

a = np.arange(2, 10, 2) # сгенерит массив от 2 до 10(не включительно) с шагом 2 - 2 4 6 8
print(a.dtype, a) # int64 [2 4 6 8]

a = np.linspace(1, 10, 4) # массив равномерно распределенных чисел(1 4 7 10 если 4)(1 3.25 5.5 7.75 10 если 5 и тд.)
print(a.dtype, a) # float64 [ 1.  4.  7. 10.]

a = np.random.rand(5) # пять случайных чисел от 0 до 1(количество - принимаемый аргумент)
print(a.dtype, a) # float64 [0.48431522 0.74445515 0.10281583 0.04371983 0.68869549]

a = np.random.rand(5, 5)
print(a.dtype, a) # матрица 5х5 элементы от 0 до 1

a = np.random.randn(3)
print(a.dtype, a) # случайные три числа из нормального распределения

a = np.random.randn(5, 5)
print(a.dtype, a) # 5x5 матрица рандомных чисел из норм. распределения

print()

array = np.array([[1, 2], [3, 4], [5, 6]])
print(array)
print(array.shape) # размер матрицы(3 х 2)
# array = array.reshape(5, 8) # не смогу сделать, потому что в матрице 6 ячеек, а нужно 40
array = array.reshape(2, 3) # делаем матрицу 3 в строке 2 в столбце(2 х 3) [[1 2 3] [4 5 6]]
print(array)

print("kek")
print()
expanded = np.expand_dims(array, axis=0) # добавили одно измерение матрице
print(expanded)
print(); print()

### СКАЛЯРНЫЕ ОПЕРАЦИИ:
newArray = np.array([[1, 1, 2], [2, 2, 3]])
array = array * 2 # умножение матрицы на число - умножение всех элементов матрицы на число
print(array) # [[1 2 3] [4 5 6]] -> [[2 4 6] [8 10 12]]

# вычитание матрицы
array -= newArray
print(array.dtype, array) # [[2 4 6] [8 10 12]] - [[1, 1, 2], [2, 2, 3]] = [[1, 3, 4], [6, 8, 9]]

# деление
# array = array / 2
# print(array) # [[0.5 1.5 2. ] [3.  4.  4.5]] поделили матрицу на 2(преобразовалась во флоты)

# возведение в степень
array = array ** 2
print(array) # [[ 1  9 16] [36 64 81]]


# срезы массивов
array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
subset = array[1:3, 0:2]  # Строки 1 и 2, столбцы 0 и 1
print(subset)
# Вывод:
# [[4, 5]
#  [7, 8]]

print()
print(array[0]) # вывод первой строки - [1 2 3]
print(array[-1]) # вывод последней строки - [7 8 9]

print(array[:, 1]) # [2 5 8] - вывод 1-го столбца

print()
print(array[0:3:2]) # [[1 2 3] [7 8 9]] 1 и 3 строки(потому что от 0 до 3 с шагом 2)

positive = array[array > 3] # выбираем все элементы массива которые больше 3
print(positive) # [4 5 6 7 8 9]
# аналогичная запись
positive = array > 3
print(array[positive])


pos = array > 3 # создаст массив из bool, где каждый элемент - результат сравнения элемента с 3(true или false)
print(pos) # [False False False] так как 1 2 3 не больше 3 [ True  True  True] [ True  True  True] так как 4 5 6 7 8 9 больше 3

# несколько условий
mask = (array > 3) & (array < 8)
print(mask) # [[False False False] [ True  True  True] [ True False False]]
print(array[mask]) # [4 5 6 7] - выведет только те элементы которые > 3 и < 8

# изменение элементов по условию
array[array < 5] = 0  # Заменяем все элементы < 5 на 0
print(array)
# Вывод: [[0, 0, 0] [0, 5, 6] [7, 8, 9]] - все элементы < 5 станут нулями

# Область применения:
# Фильтрация тренировочных данных: Например, выбор всех образцов с определенным классом (labels[labels == 1]).
# Удаление выбросов: data[data < np.percentile(data, 95)] для исключения экстремальных значений


array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
# rows = array[[0, 2], :]  # Выбираем строки 0 и 2
# эквивалентная запись
rows = array[[0, 2]]  # Выбираем строки 0 и 2
print(rows) # [[1 2 3] [7 8 9]]


elements = array[[0, 1, 2], [2, 1, 0]]  # Элементы (0,2), (1,1), (2,0)
print(elements)  # Вывод: [3, 5, 7]

cols = array[:, [0, 2]]  # Столбцы 0 и 2
print(cols)
print(cols.reshape(2, 3)) # делаем матрицу 3 на 2 матрицей 2 на 3 - [[1 3 4] [6 7 9]]
# Вывод: [[1, 3] [4, 6] [7, 9]] - колонки 1 4 7

print()

indices = np.array([0, 2])
# используем матрицу для подстановки в функцию нарезания матрицы
subset = array[indices, :]  # То же, что array[[0, 2], :]
print(subset)

indices = np.random.choice(len(array), size=12, replace=True)
print("indices:", indices)
batch = array[indices] # рандомный набор индексов [0 1 2 2 1 0 1 2 1 2 1 1] от 0 до 2 (12 элементов)
print(batch.shape) # 12 x 3 матрица
print(batch) # [[4 5 6] [7 8 9] [1 2 3] [1 2 3] [1 2 3] [1 2 3] [1 2 3] [4 5 6] [7 8 9] [7 8 9] [1 2 3] [4 5 6]]
             # рандомно вставила строки в новую матрицу размером size x размерСтроки
# пример с replace=False:
idices = np.random.choice(len(array), size=2, replace=False)
print("indexes:", idices) # [2 0] - рандомный набор индексов от 0 до 2 (2 элемента)
batch = array[idices] # аналогично array[[2, 0]] - выбор 2 и 0 строки массива array -  [[7 8 9] [1 2 3]]
print(batch.shape)
print(batch)
