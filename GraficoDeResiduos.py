import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Cargar el dataset "mpg"
mpg = sns.load_dataset('mpg').dropna()  # Cargamos el dataset y eliminamos filas con valores faltantes
X = mpg[['horsepower', 'weight', 'displacement']].values  # Seleccionamos algunas características tecnológicas
y = mpg['mpg'].values  # Millas por galón (rendimiento del combustible)

"""
 *) horsepower (caballos de fuerza), weight (peso del vehículo) y displacement (cilindrada del motor).

 *) El galón hace referencia al volumen de un líquido, por lo general combustible, vino o cerveza. De acuerdo al detalle 
 del diccionario de la Real Academia Española (RAE), 
 en Norteamérica un galón es equivalente a 3,785 litros (también en España).
 En Gran Bretaña, en cambio el galón equivale a 4,546 litros.

 *) una milla equivale a 1 kilómetro y 609 metros.
 """

# 2. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Ajustar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predecir en el conjunto de prueba y calcular los residuos
y_pred = model.predict(X_test)
residuos = y_test - y_pred

# 5. Visualizar los residuos utilizando un gráfico de dispersión
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuos, color='blue', edgecolor='k')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)  # Línea horizontal en 0
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.title('Gráfico de Residuos - Dataset MPG')
plt.show()
