
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense


def create_model(trn_sizes):

    from tensorflow.keras import layers

    vadModel = tf.keras.Sequential()
    vadModel.add(layers.LSTM(trn_sizes.LSTMCell))
    vadModel.add(Dense(10, activation='sigmoid'))
    vadModel.add(Dense(1, activation='sigmoid'))
    vadModel.compile(optimizer='adam', loss='mse')
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return vadModel