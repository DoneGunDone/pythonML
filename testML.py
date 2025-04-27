import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
# НЕТ ЭТОГО ДАТАСЕТА => не обучить
df = pd.read_csv("house_prices.csv")

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Создание и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train) # обучение на тренировочных данных

# Предсказание на тестовых данных
y_pred = model.predict(X_test)
print(y_pred[:10]) # выведет предсказанные стоимости первых 10 домов из тестовой выборки

# Оценка модели
mse = mean_squared_error(y_test, y_pred) # получим большое число - около 4895948726.587
rmse = np.sqrt(mse)  # Вычисление RMSE - корень = 69971.06 примерно 70к средняя ошибка будет
r2 = r2_score(y_test, y_pred)

# Вывод результатов
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")