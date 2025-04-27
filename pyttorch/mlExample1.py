import torch
import torch.optim as optim
import torch.nn as nn

# Определяем устройство с правильной логикой
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Данные
# X=[[0,0,1],[1,1,1],[1,0,0],[0,1,0]] (4 примера: разные комбинации признаков).
# Y=[1,0,0,0] (правильные ответы: нравится только партнёр без квартиры, без любви к року, но привлекательный).

# Данные (создаём сразу на нужном устройстве)
X = torch.tensor([[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float32, device=device)
Y = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=device)


# Определяем сеть
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Используем nn.Linear для автоматического добавления bias
        self.hidden = nn.Linear(3, 2)  # 3 входа -> 2 нейрона в скрытом слое
        self.output = nn.Linear(2, 1)  # 2 нейрона -> 1 выход

        # Инициализируем веса как в исходной сети
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]]))
            self.output.weight.copy_(torch.tensor([[-1.0, 1.0]]))

    def forward(self, x):
        Zh = self.hidden(x)
        Uh = torch.sigmoid(Zh)
        Zout = self.output(Uh)
        return Zout  # Возвращаем logits (без сигмоиды, BCEWithLogitsLoss сделает это сам)

# Создаём модель
model = SimpleNet().to(device)

# Оптимизатор (попробуем Adam вместо SGD)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Функция потерь (бинарная кросс-энтропия с logits)
criterion = nn.BCEWithLogitsLoss()

# Обучение
num_epochs = 500  # Увеличим количество эпох
for epoch in range(num_epochs):
    # Прямой проход
    logits = model(X).squeeze()  # Получаем logits
    loss = criterion(logits, Y)  # Считаем ошибку

    # Обратное распространение
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Итоговые результаты
with torch.no_grad():
    logits = model(X).squeeze()
    Y_pred = (torch.sigmoid(logits) >= 0.5).float()
print(f"Final W1:\n{model.hidden.weight}")
print(f"Final W2:\n{model.output.weight}")
print(f"Predictions: {Y_pred}")
print(f"True labels: {Y}")