import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle
import cv2, sys

# filename = sys.argv[0]

image_shape = (28, 28, 1)
# x = cv2.imread(filename)
x = cv2.imread('6.png')
x = cv2.resize(x, (28,28), interpolation = cv2.INTER_AREA)
print(x.shape)
x = x.astype('float32') / 255.0
x  = x.reshape((-1,) + image_shape)

model = tf.keras.models.load_model('modelv2')

y_pred = model.predict(x)
print('The number is :', np.argmax(y_pred))
