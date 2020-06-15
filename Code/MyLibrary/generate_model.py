# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:25:40 2020

@author: omer
"""
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import Adam



class Create_model:
    
    def __init__(self):
        self.model = Sequential()
        
    def add_convolutional_layers(self, num_layers, input_shape):
        self.model.add(Conv2D(32, (3, 3), input_shape = input_shape, activation = 'relu'))
        self.model.add(MaxPooling2D(pool_size = (2, 2)))
        for repeat in range(1,num_layers):
            self.model.add(Conv2D(32, (3, 3), activation = 'relu'))
            self.model.add(MaxPooling2D(pool_size = (2, 2)))
            
    def flatten_model(self):
        self.model.add(Flatten())
    
    def add_neural_network_layers(self, cells_list, operation_list):
        for cells, operation in zip(cells_list, operation_list):
            self.model.add(Dense(units = cells, activation = operation))
            
    def compile_model(self, learning_rate):
        self.model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        
    def get_model_structure(self):
        return self.model
    
    
    
        
        
    