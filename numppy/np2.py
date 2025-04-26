import numpy as np

a1 = np.array([1, 2, 3, 4]) # массив int64
a = np.array([1, 2, 3, 4], 'float64')
print(a.dtype, a) # float64 [1. 2. 3. 4.]

print(np.sctypeDict) # все алиасы для типов данных
# {'bool': <class 'numpy.bool'>, 'float16': <class 'numpy.float16'>, 'float32': <class 'numpy.float32'>, 'float64': <class 'numpy.float64'>, 'longdouble': <class 'numpy.longdouble'>, 'complex64': <class 'numpy.complex64'>, 'complex128': <class 'numpy.complex128'>, 'clongdouble': <class 'numpy.clongdouble'>, 'bytes_': <class 'numpy.bytes_'>, 'str_': <class 'numpy.str_'>, 'void': <class 'numpy.void'>, 'object_': <class 'numpy.object_'>, 'datetime64': <class 'numpy.datetime64'>, 'timedelta64': <class 'numpy.timedelta64'>, 'int8': <class 'numpy.int8'>, 'byte': <class 'numpy.int8'>, 'uint8': <class 'numpy.uint8'>, 'ubyte': <class 'numpy.uint8'>, 'int16': <class 'numpy.int16'>, 'short': <class 'numpy.int16'>, 'uint16': <class 'numpy.uint16'>, 'ushort': <class 'numpy.uint16'>, 'int32': <class 'numpy.int32'>, 'intc': <class 'numpy.int32'>, 'uint32': <class 'numpy.uint32'>, 'uintc': <class 'numpy.uint32'>, 'int64': <class 'numpy.int64'>, 'long': <class 'numpy.int64'>, 'uint64': <class 'numpy.uint64'>, 'ulong': <class 'numpy.uint64'>, 'longlong': <class 'numpy.longlong'>, 'ulonglong': <class 'numpy.ulonglong'>, 'intp': <class 'numpy.int64'>, 'uintp': <class 'numpy.uint64'>, 'double': <class 'numpy.float64'>, 'cdouble': <class 'numpy.complex128'>, 'single': <class 'numpy.float32'>, 'csingle': <class 'numpy.complex64'>, 'half': <class 'numpy.float16'>, 'bool_': <class 'numpy.bool'>, 'int_': <class 'numpy.int64'>, 'uint': <class 'numpy.uint64'>, 'float': <class 'numpy.float64'>, 'complex': <class 'numpy.complex128'>, 'object': <class 'numpy.object_'>, 'bytes': <class 'numpy.bytes_'>, 'a': <class 'numpy.bytes_'>, 'int': <class 'numpy.int64'>, 'str': <class 'numpy.str_'>, 'unicode': <class 'numpy.str_'>}

a = np.array([1, 2, 3, 4], 'uintc')
print(a.dtype, a) # uint32 [1 2 3 4]
a = np.array([1, 2, 3, 4], 'intc')
print(a.dtype, a) # int32 [1 2 3 4]

a = np.array([1, 2, 3, 4], 'str_')
print(a.dtype, a) # <U1 ['1' '2' '3' '4']

k = np.complex64(10)
print(k.dtype, k) # complex64 (10+0j)

c = np.int16(10.5)
print(c.dtype, c) # int16 10

# a = np.array([1, 2, 5000, 800], 'int8')
# print(a.dtype, a) # ошибка 5000 out of bounds for int8

a = np.array([1, 2, 5000, 800])
print(a.dtype, a) # int64 [   1    2 5000  800] автоматом подставилось int64

a = np.array([1, 2, 5000, 800], 'int16')
print(a.dtype, a) # int16 [   1    2 5000  800] - int16 хватает на такой массив

a = np.complex64(a)
print(a.dtype, a) # complex64 [1.e+00+0.j 2.e+00+0.j 5.e+03+0.j 8.e+02+0.j] преобразовали тип

b = np.complex64(a)
print(b.dtype, b)

# c = np.int32(b) # так не надо делать от частных(complex) к общим(int) типам данных, возможна потеря данных
# print(c.dtype, c) # ComplexWarning: Casting complex values to real discards the imaginary part

# k = np.float32(b) # ComplexWarning: Casting complex values to real discards the imaginary part
# print(k.dtype, k) # возможна потеря данных - float32 [1.e+00 2.e+00 5.e+03 8.e+02]

print(np.array((1, 2, 3))) # кортеж преобразуется в массив [1 2 3]

z = np.array("Hello")
print(z.dtype, z) # <U5 Hello один элемент в массиве - строка Hello

# массив массивов(матрица):
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a.dtype, a)

# неправильный массив массивов
# a = np.array([[1, 2], [3, 4], [5, 6, 7]]) # ошибка setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (3,) + inhomogeneous part

# трехмерный массив
h = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
print(h.dtype, h)
# первая ось - слева направо
# вторая ось - сверху вниз
# третья ось - спереди назад

print(h[1][0][1]) # 6
