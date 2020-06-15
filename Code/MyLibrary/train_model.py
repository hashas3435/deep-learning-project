# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:25:59 2020

@author: omer
"""

import matplotlib.pyplot as plt


class Fit_model: 
    
    def __init__(self, model):
        self.model = model
        self.history = None
        
        
    def run(self, train_batches, valid_batches, train_steps, valid_steps, epochs):
        self.history = self.model.fit_generator(generator=train_batches,
                    steps_per_epoch= train_steps, 
                    validation_data=valid_batches,
                    validation_steps= valid_steps, 
                    epochs = epochs, verbose=1)        
        
        
    def plot_history(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        