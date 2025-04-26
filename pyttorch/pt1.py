import torch
import numpy as np
# print(torch.__version__)

# t = torch.Tensor([1, 2, 3]) # создаст тензор флотов 32
# print(t, t.dtype) # tensor([1., 2., 3.]) torch.float32
# t = torch.empty(3, 5, 2, dtype=torch.int8)
# print(t.shape, t)
#
# t = torch.tensor([1, 2, 3]) # создаст тензор интов 64
# print(t, t.dtype) # tensor([1, 2, 3]) torch.int64

# Типы данных PyTorch:
# torch.HalfTensor   -   16 бит, с плавающей точкой
# torch.FloatTensor  -   32 бита, с плавающей точкой
# torch.DoubleTensor -   64 бита, с плавающей точкой
# torch.ShortTensor  -   16 бит, целочисленный, знаковый
# torch.IntTensor    -   32 бита, целочисленный, знаковый
# torch.LongTensor   -   64 бита, целочисленный, знаковый
# torch.CharTensor   -   8 бит, целочисленный, знаковый
# torch.ByteTensor   -   8 бит, целочисленный, беззнаковый
# torch.BoolTensor   -   булевый (True/False)

# t = torch.ByteTensor([1, 2, 3]) # dtype сюда нельзя прописать(тип заранее определен - uint8
# print(t) # tensor([1, 2, 3], dtype=torch.uint8)
#
t = torch.DoubleTensor(3, 2, 3, 5)
# print(t, t.dtype, t.shape) #  dtype=torch.float64) torch.float64 torch.Size([3, 2, 3, 5])

print(t.type()) # вернется тип тензора - torch.DoubleTensor

# dim() - вернет количество осей тензора
print(t.dim()) # 4

# size() - вернет количество элементов на каждой оси - torch.Size([3, 2, 3, 5])
print(t.size())


### преобразование np массивов в тензоры и обратно
dnp = np.array([[1, 2, 3,],  [4, 5, 6]]) # int64 массив 2х3
print(dnp.shape)

t2 = torch.from_numpy(dnp)
print(t2.type(), t2, t2.dtype) # torch.LongTensor tensor([[1, 2, 3], [4, 5, 6]]) torch.int64

t3 = torch.tensor(dnp, dtype=torch.float32)
print(t3.type(), t3, t3.dtype) # torch.DoubleTensor tensor([[1., 2., 3.], [4., 5., 6.]]) torch.float32

# если изменить dnp, то t2 изменится(так как тип не менялся), а t3 не изменится(тип другой - копия, а не ссылка)
dnp[0][0] = 5
print(t2[0][0]) # tensor(5)
print(t3[0][0]) # tensor(1.)

t2[0][0] = 1
print(dnp[0][0]) # поменялся на 1

# тензоры и списки не связаны ссылками, так как при создании от списка тензор копируется
spisok = [[1, 2, 3], [4, 5, 6]]
tsp = torch.tensor(spisok)
tsp[0][0] = 5
print(spisok[0][0]) # 1

spisok[0][1] = 22
print(tsp[0][1]) # tensor(2) => не поменялся(должен был поменяться на 22)


# преобразование тензора в numpy массив:
d = t3.numpy()
print(d, d.shape, d.dtype) #  [[1. 2. 3.] [4. 5. 6.]] (2, 3) float32

# Методы преобразования типа в PyTorch:
# half()   -   torch.float16
# float()  -   torch.float32
# double() -   torch.float64
# short()  -   torch.int16
# int()    -   torch.int32
# long()   -   torch.int64
# char()   -   torch.int8
# byte()   -   torch.uint8
# bool()   -   torch.bool

# если размерность не теряется, то тензоры сохраняют связь по ссылке:
t = torch.tensor([1, 2, 3])
print(t, t.shape)

tt = t.long()
print(tt, tt.shape)

tt[0]  = 8
print(tt, t) # tensor([8, 2, 3]) tensor([8, 2, 3])


