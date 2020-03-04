from __future__ import absolute_import, division, print_function, unicode_literals
import math
import tensorflow as tf



print() # used to seperate the unclearable errors
print("Tensorflow Version:")
print(tf.__version__)
print()

mnist = tf.keras.datasets.mnist # pulling installed mnist dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([ # Defining our model
  tf.keras.layers.Flatten(input_shape=(28, 28)), 
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

