# Importar las bibliotecas necesarias
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Cargar y preparar el conjunto de datos MNIST
(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = mnist.load_data()

# Preprocesamiento de los datos de entrenamiento y prueba
x_entrenamiento = x_entrenamiento.reshape(-1, 28, 28, 1)
x_prueba = x_prueba.reshape(-1, 28, 28, 1)
x_entrenamiento = x_entrenamiento.astype('float32') / 255
x_prueba = x_prueba.astype('float32') / 255
y_entrenamiento = to_categorical(y_entrenamiento, num_classes=10)
y_prueba = to_categorical(y_prueba, num_classes=10)

# Crear el modelo de la red neuronal
modelo = Sequential()
modelo.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo.add(MaxPooling2D(pool_size=(2, 2)))
modelo.add(Flatten())
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(10, activation='softmax'))

# Compilar el modelo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(x_entrenamiento, y_entrenamiento, batch_size=128, epochs=10, verbose=1, validation_data=(x_prueba, y_prueba))

# Evaluar el modelo en el conjunto de prueba
puntaje = modelo.evaluate(x_prueba, y_prueba, verbose=0)
print('Pérdida en el conjunto de prueba:', puntaje[0])
print('Precisión en el conjunto de prueba:', puntaje[1])