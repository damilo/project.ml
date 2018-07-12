### CLASS MLPQNet
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K


class MLPQNet ():


    def __init__ (self, input_shape, num_outputs, lr=0.01):
        # input layer
        inputs = layers.Input (shape=input_shape, name='input')
        # hidden layer
        net = layers.Dense (64, activation='relu', name='hidden_1') (inputs)
        net = layers.Dense (48, activation='relu', name='hidden_2') (net)
        net = layers.Dense (48, activation='relu', name='hidden_3') (net)
        net = layers.Dense (24, activation='relu', kernel_regularizer=self.__l2_reg, name='hidden_4') (net)
        # output layer
        outputs = layers.Dense (num_outputs, activation='linear', name='output') (net)

        # create optimizer
        opt_sgd = optimizers.SGD (lr=lr)
        opt_adam = optimizers.Adam (lr=lr)

        # build and compile model
        self.model = models.Model (inputs=inputs, outputs=outputs)
        self.model.compile (optimizer=opt_adam, loss=self.__mean_squared_error)
        
        # save some stuff internally for reset
        self.__lr = lr


    # declaration of loss function shall only remind on how target update works in function approximation
    def __mean_squared_error (self, target_values, approximation_values):
        return K.mean (K.square (target_values - approximation_values), axis=-1)
    
    
    def __l2_reg (self, weight_matrix):
        __lambda = 0.07
        return __lambda * K.sum (K.square (weight_matrix))