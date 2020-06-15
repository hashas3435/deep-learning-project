# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:14:43 2020

@author: omer
"""

from keras.preprocessing.image import ImageDataGenerator

class Generate_data:
    
    def __init__(self, train_path, valid_path, test_path, image_size): # image_size is a tuple resembling the length and width of all the images in test, valid and test data 
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.image_size = image_size
        
        self.train_form  = None
        self.valid_form = None
        self.test_form = None
        self.train_batches = None
        self.valid_batches = None
        self.test_batches = None
        
        
        
    
    def reform_image(self, given_shear_range = 0.2, given_zoom_range = 0.2, 
                given_horizontal_flip = True):
        
        self.train_form = ImageDataGenerator(rescale = 1./255,
                                   shear_range = given_shear_range,
                                   zoom_range = given_zoom_range,
                                   horizontal_flip = given_horizontal_flip)    # designed to be more complicated than test, in order to give the computer many possible scenarios  
        self.valid_form = ImageDataGenerator(rescale = 1./255)
        self.test_form = ImageDataGenerator(rescale = 1./255)
        
    def create_batches(self, train_batch_size, valid_batch_size, test_batch_size, categories):
        self.train_batches = self.train_form.flow_from_directory(self.train_path,
                                                 target_size =  self.image_size,
                                                 batch_size = train_batch_size,
                                                 classes = categories)
        
        self.valid_batches = self.valid_form.flow_from_directory(directory=self.valid_path,
                                    target_size= self.image_size,
                                     batch_size= valid_batch_size,
                                     classes = categories)
        
        self.test_batches = self.test_form.flow_from_directory(self.test_path,
                                            target_size = self.image_size,
                                            batch_size = test_batch_size,
                                            classes = categories) 
    
    def get_train_batches(self):
        return self.train_batches
    def get_valid_batches(self):
        return self.valid_batches
    def get_test_batches(self):
        return self.test_batches
    def get_all_batches(self):
        return (self.train_batches, self.valid_batches,  self.test_batches)