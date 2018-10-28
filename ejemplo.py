# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow import keras


# Helper libraries
import numpy as np

# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# print(tf.__version__)

# Descargamos los datos
fashion_mnist = keras.datasets.fashion_mnist

# Cargamos los valoes
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# Definimos las clases
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Escalamos los valores a un rango entre 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


# Construimos el modelo
model = keras.Sequential([
    # La primera capa representa los 784 pixeles
    keras.layers.Flatten(input_shape=(28, 28)),
    # Esta capa esta compuesta por 128 neuronas
    keras.layers.Dense(128, activation=tf.nn.relu),
    # Ultima capa compuesta por 10 neuronas las caules representan una de las clases
    # retorna un arreglo compuesto por 10 probabilidades, cada una representa la
    # probabilidad de pertenecer auna de las neuronas la mas alta es la solucion
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),       # Esta funcion describe como el modelo sera actualizado en funcion de la funcion
              # Esto define la forma en la que se buscara minimizar las salidas
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])                     # La metrica se basa en la precision que tenga cada clasificacion

tensorboard = TensorBoard(
    log_dir="logs/{}", histogram_freq=0, write_graph=True, write_images=True)

# 5 Epocas
model.fit(train_images, train_labels, epochs=5,
          verbose=1, callbacks=[tensorboard])

# Calculamos la precision contra los datos correctos
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols, 2*i+1)
#     plot_image(i, predictions, test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions, test_labels)
# plt.show()


# Grab an image from the test dataset
img = test_images[0]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

plt.show()
