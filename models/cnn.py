import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential, Model


def CNN(input_shape, num_classes=2):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='elu', input_shape=input_shape))
    model.add(Conv1D(filters=16, kernel_size=2, padding='same', activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='elu'))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='elu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save("cnn.h5")