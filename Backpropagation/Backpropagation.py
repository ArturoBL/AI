import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Crear datos de entrenamiento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Crear modelo de red neuronal
modelo = Sequential()
modelo.add(Dense(units=2, input_dim=2, activation='sigmoid'))
modelo.add(Dense(units=1, activation='sigmoid'))

# Compilar modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelo
historia_entrenamiento = modelo.fit(X, y, epochs=5000, batch_size=4)

modelo.save('modelo_entrenado.h5')

# Evaluar modelo
resultados = modelo.evaluate(X, y)
print('Loss:', resultados[0])
print('Accuracy:', resultados[1])

# Hacer predicciones
predicciones = modelo.predict(X)
print('Predicciones:', predicciones.round())

v = [[0,0]]
print(f'Prediccion {v}: {modelo.predict(v)}')

v = [[0,1]]
print(f'Prediccion {v}: {modelo.predict(v)}')

v = [[1,0]]
print(f'Prediccion {v}: {modelo.predict(v)}')

v = [[1,1]]
print(f'Prediccion {v}: {modelo.predict(v)}')


# Gráfico de pérdida (loss)
plt.plot(historia_entrenamiento.history['loss'])
plt.title('Pérdida (Loss) durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()

# Gráfico de precisión (accuracy)
plt.plot(historia_entrenamiento.history['accuracy'])
plt.title('Precisión (Accuracy) durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.show()