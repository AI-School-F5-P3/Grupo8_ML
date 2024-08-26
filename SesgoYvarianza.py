import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Cargar el dataset "California Housing"
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Inicializar listas para almacenar errores
train_errors = []
val_errors = []
degrees = range(1, 15)

# 4. Entrenar modelos con diferentes grados de polinomios
for degree in degrees:
    # Transformar los datos para incluir características polinómicas
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Ajustar el modelo
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predecir y calcular errores
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_test, y_test_pred))

# 5. Visualizar la curva de validación
plt.figure(figsize=(10, 6))
plt.plot(degrees, train_errors, label='Error de Entrenamiento', marker='o')
plt.plot(degrees, val_errors, label='Error de Validación', marker='o')
plt.xlabel('Grado del Polinomio')
plt.ylabel('Error Cuadrático Medio')
plt.title('Curva de Validación: Sesgo vs. Varianza')
plt.legend()
plt.yscale('log')  # Usar escala logarítmica para mejorar la visualización
plt.grid(True)
plt.show()
