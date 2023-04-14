# Importar las bibliotecas necesarias
import numpy as np
from nnlib import NeuralNetwork

# Crear datos de entrenamiento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Crear una instancia de la red neuronal
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 5000

# Crear una instancia de la red neuronal
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Entrenar la red neuronal
nn.train(X, y, epochs, learning_rate)

# Hacer predicciones
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.predict(X_test)

# Imprimir las predicciones
print("Predicciones:")
for i in range(X_test.shape[0]):
    print(f'Entrada: {X_test[i]}, Predicción: {predictions[i][0]}')