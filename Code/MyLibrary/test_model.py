# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:36:27 2020

@author: omer
"""

import numpy as np

class Classify:
    
    def __init__(self, model, categories):
       self.model = model
       self.categories = categories
       self.right_results = np.zeros(len(self.categories))
       self.wrong_results = np.zeros(len(self.categories))
       self.test_labels = None
       self.predictions_list = None
    
    def test(self, test_batches):
        test_imgs, self.test_labels = next(test_batches)
        self.predictions_list = self.model.predict_generator(generator=test_batches, steps=1, verbose=0, workers = 1)
        self.interpret_predictions()
        
    
    def interpret_predictions(self):
        for prediction, label in zip(self.predictions_list, self.test_labels):
            if np.argmax(prediction) == np.argmax(label):
                self.right_results[np.argmax(label)] += 1
            else:
                self.wrong_results[np.argmax(label)] += 1
    
    def print_results(self):
        for category, right , wrong in zip(self.categories, self.right_results, self.wrong_results):
            print ("predicted {} pictures right: {}".format(category, right))
            print ("predicted {} pictures wrong: {}".format(category, wrong))
            print (r"Succcess rate for {} predictions: {}%".format(category, self.get_rate(right, right + wrong)))
        print (r"Total success rate: {}%".format(self.get_rate(np.sum(self.right_results), len(self.predictions_list))))
                   
                   
    def get_rate(self, right, total):
        return int(100*round(right/total, 2))
    
    def get_wrong_results(self):
        return self.wrong_results
    def get_right_results(self):
        return self.right_results
    
        
