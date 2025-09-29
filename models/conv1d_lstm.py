import tensorflow as tf
import keras
from tensorflow.keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, TimeDistributed, Flatten, LSTM
from tensorflow.keras.models import Sequential, Model

def CNN_LSTM(input_shape, num_classes=2):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=16, kernel_size=2, padding='same', activation='elu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=2, padding='same', activation='elu')))
    model.add(TimeDistributed(Conv1D(filters=4, kernel_size=2, padding='same', activation='elu')))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save("cnn-lstm.h5")