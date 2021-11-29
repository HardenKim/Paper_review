

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

class DNN(Layer):
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0., **kwargs):
        """
        
        """
        super(DNN, self).__init__(**kwargs)
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)
        
    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
    