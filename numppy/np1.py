import numpy as np

# (0, 1) 0 - индекс по вертикали, 1 - индекс по горизонтали в матрице

# создание массива в numpy
myarr = np.array([1, 2, 3])

print(type(myarr), myarr) # <class 'numpy.ndarray'> [1 2 3]
print(myarr.dtype) # int64
a = np.array([2, True, "kek"])

print(a) # все приводится к строкам - ['2' 'True' 'kek'] потому что тип должен быть один

print(a[0])
a[1] = 123
print(a) # ['2' '123' 'kek'] 123 автоматом парсится в строку

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(a[2])

a2 = a[[1, 1, 1, 1, 1, 1, 1]] # создал массив из семи ПЕРВЫХ ИНДЕКСОВ массива а - из семи двоек
print(a2) # [2 2 2 2 2 2 2]

a3 = a[ [True, False, True, True, False, True, True, False, False] ]
print(a3) # [1 3 4 6] - true под соответствующим индексом(первый true - 1), второй false - не вносится, 3 true - 3 есть
# 4 true - есть, 5 false - нет, 6 true - есть, 7 True - есть, 8 нет, 9 нет

b = a.reshape(3, 3) # [[1 2 3] [4 5 6] [7 8 9]] матрица
print(b)
print(b[0][1]) # 2