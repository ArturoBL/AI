from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Cargar modelo desde archivo
modelo_cargado = load_model('modelo_entrenado.h5')
print('Modelo cargado exitosamente.')

# Crear datos de entrenamiento
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
# Historia de entrenamiento del modelo
historia_entrenamiento = modelo_cargado.fit(X, y, epochs=5000, batch_size=4, verbose=0)


# Hacer predicciones con el modelo cargado
v = [[1,0]]
print(f'Prediccion {v}: {modelo_cargado.predict(v)}')


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