from keras.models import load_model
import numpy as np


# Cargar modelo desde archivo
modelo_cargado = load_model('modelo_entrenado.h5')
print('Modelo cargado exitosamente.')

# Hacer predicciones con el modelo cargado
v = [[1,0]]
print(f'Prediccion {v}: {modelo_cargado.predict(v)}')

v = [[0,0]]
print(f'Prediccion {v}: {modelo_cargado.predict(v)}')