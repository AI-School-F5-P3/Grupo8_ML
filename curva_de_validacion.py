import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
grados = [1, 2, 3, 4]  # Grado del polinomio (complejidad del modelo)
error_entrenamiento = [5000, 3000, 2000, 1500]  # Error de entrenamiento
error_validacion = [5200, 3500, 4000, 5000]  # Error de validación

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Graficar error de entrenamiento
plt.plot(grados, error_entrenamiento, marker='o', label='Error de Entrenamiento', color='blue')

# Graficar error de validación
plt.plot(grados, error_validacion, marker='o', label='Error de Validación', color='red')

# Añadir etiquetas y título
plt.xlabel('Complejidad del Modelo (Grado del Polinomio)')
plt.ylabel('Error')
plt.title('Curva de Validación')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()
