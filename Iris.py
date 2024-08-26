from sklearn.datasets import load_iris

# Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas de clase

# Imprimir las primeras filas de los datos
print("Características:\n", X[:5])
print("Etiquetas:\n", y[:5])
