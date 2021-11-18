"""
model : Factorization Machines (FM)

Referenece : ZiyaoGeng/Recommender-System-with-TF2.0
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2

class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """[summary]

        Args:
            feature_columns (list): sparse column feature information
            k (int): the latent vector
            w_reg (float, optional): the regularization coefficient of parameter w. Defaults to 1e-6.
            v_reg (float, optional): the regularization coefficient of parameter V. Defaults to 1e-6.
        
        """
        super(FM_Layer, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.index_mapping = []
        self.feature_length = 0
        for feature in self.sparse_feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feature['feature_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        
    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                initializer=tf.zeros_initializer(),
                                trainable=True)
        self.w0 = self.add_weight(name='w0', shape=(1,),)
        # self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
        #                           initializer=tf.zeros_initializer(),
        #                           regularizer=l2(self.w_reg),
        #                           trainable=True)
        # self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
        #                           initializer=tf.zeros_initializer(),
        #                           regularizer=l2(self.v_reg),
        #                           trainable=True)
    
    def call(self, inputs, **kwargs):
        # mapping
        inputs = inputs + tf.convert_to_tensor(self.index_mapping)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1) # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs) # (batch_size, feature_length, k)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True)) # (batch_size, 1, k)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True) # (batch_size, 1, k)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2) # (batch_size, 1)
        #output
        ouputs = first_order + second_order
        return ouputs

    
class FM(Model):
    def __init__(self, feature_columns, k, w_reg=1e-6, v_reg=1e-6):
        """[summary]

        Args:
            feature_columns (list): sparse column feature information
            k (int): the latent vector
            w_reg (float, optional): the regularization coefficient of parameter w. Defaults to 1e-6.
            v_reg (float, optional): the regularization coefficient of parameter V. Defaults to 1e-6.
        """
        super(FM, self).__init__()
        self.sparse_feature_columns = feature_columns
        self.fm = FM_Layer(feature_columns, k, w_reg, v_reg)
        
    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs
    
    def summary(self, **kwargs):
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        Model(inputs=sparse_inputs, outputs=self.call(sparse_inputs)).summary()
        
        