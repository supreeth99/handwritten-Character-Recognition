import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import pickle 
# from PIL import Image

def build_model_acc(input_shape, num_classes):
    x = Input(input_shape)
    y = Conv2D(8,  (3, 3),padding="same", activation='relu')(x)
    y = MaxPooling2D((2,2))(y)
    y = Conv2D(16, (3, 3),padding="same", activation='relu')(y)
    y = MaxPooling2D((2,2))(y)
    y = Flatten()(y)
    y = Dense(128, activation="relu")(y)
    y = Dense(64, activation="relu")(y)
    y = Dense(num_classes, activation="softmax")(y)
    return Model(x,y)


def build_model_rob(input_shape, num_classes):
    x = Input(input_shape)
    y = Conv2D(8,  (3, 3),padding="same", activation='sigmoid')(x)
    y = MaxPooling2D((2,2))(y)
    y = Conv2D(16, (3, 3),padding="same", activation='sigmoid')(y)
    y = MaxPooling2D((2,2))(y)
    y = Conv2D(64, (3, 3),padding="same", activation='sigmoid')(y)
    y = MaxPooling2D((2,2))(y)
    y = Flatten()(y)
    y = Dense(128, activation="sigmoid")(y)
    y = Dense(64, activation="relu")(y)
    y = Dense(num_classes, activation="softmax")(y)
    return Model(x,y)
    

batch_size = 8
num_classes = 10
epochs = 50
image_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

x_train = x_train.reshape((-1,) + image_shape)
x_test  = x_test.reshape((-1,) + image_shape)

# X_train = tf.keras.utils.normalize(X_train, axis=1)
# X_test = tf.keras.utils.normalize(X_test, axis=1)

print(x_train.shape)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

model = build_model_acc(image_shape, num_classes)
model.summary()
model.compile(
    optimizer=Adam(lr=1e-4),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)
# pickle.dump(model, open('model5.pkl', 'wb')

model.fit(
    x_train,y_train,
    epochs=epochs,
    # validation_data=(x_test,y_test)
)

model.save('modelv2')

