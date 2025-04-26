import torch
import torch.nn as nn
import torch.optim as optim
# print(torch.backends.mps.is_available()) # Должно вернуть True
#
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Простая модель
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


model = SimpleNet().to("mps")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
for epoch in range(100):
    inputs = torch.randn(32, 10).to("mps")
    targets = torch.randn(32, 1).to("mps")

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")