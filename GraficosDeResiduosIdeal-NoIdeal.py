import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generar datos simulados
np.random.seed(42)
X = np.random.normal(size=100)
y = 2 * X + np.random.normal(size=100)  # Relación lineal

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Ajustar modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predecir y calcular residuos
y_pred = model.predict(X_test)
residuos = y_test - y_pred

# Crear el gráfico de residuos ideal
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuos, color='blue', edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos Ideal')

# Generar datos con un patrón no lineal
y_non_linear = 2 * X**2 + np.random.normal(size=100)  # Relación cuadrática
X_train, X_test, y_train, y_test = train_test_split(X, y_non_linear, test_size=0.3, random_state=42)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Ajustar modelo de regresión lineal a datos no lineales
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuos = y_test - y_pred

# Crear el gráfico de residuos no ideal
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuos, color='blue', edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos No Ideal')

plt.tight_layout()
plt.show()
