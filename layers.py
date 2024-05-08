# Custom L1 DIstance layer module
# it is needed to load the custom model
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer


# custome L1 distance layer
class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
