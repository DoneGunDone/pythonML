import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Проверяем доступность устройств
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}") # шрек шлет меня нахуй


def act(X):
    return 0 if X < 0.5 else 1


def go(house, rock, attr):
    X = torch.tensor([house, rock, attr], dtype=torch.float32)
    Wh = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])  # матрица 2x3
    Wout = torch.tensor([[-1.0, 1.0]])  # вектор 1x2

    Zh = torch.mv(Wh, X)  # вычисляем сумму на входах нейронов скрытого слоя
    print(f"начальная сумма на входах нейронов скрытого слоя: {Zh}")

    Uh = torch.tensor([act(x) for x in Zh], dtype=torch.float32)
    print(f"значения на выходах нейронов скрытого слоя: {Uh}")

    Zout = torch.dot(Wout, Uh)
    Y = act(Zout)
    print(f"выходное значение НС: {Y}")

    return Y

house, rock, attr = 1, 1, 1

res = go(house, rock, attr)
print(f'result: {res}')