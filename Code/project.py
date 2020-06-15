# -*- coding: utf-8 -*-
"""
Created on Wed May 13 15:54:26 2020

@author: omer
"""

from MyLibrary.examine_path import Path_Detector
from  MyLibrary.prepare_data import Generate_data
from MyLibrary.generate_model import Create_model
from  MyLibrary.train_model import Fit_model
from  MyLibrary.test_model import Classify




categories = ("car", "motorcycle")

train_batch_size, valid_batch_size = 32, 32
image_size = (64,64)
input_shape = image_size + (3,)
epochs = 1
learning_rate = 0.001 

units = (128, len(categories))
activation = ('relu', 'sigmoid')



def get_paths():
    input_paths = Path_Detector(categories)
    input_paths.receive_data_repository_paths()
    return input_paths


def make_data(train_path, valid_path, test_path, test_images_amount):
    data = Generate_data(train_path, valid_path, test_path, image_size)
    data.reform_image()
    data.create_batches(train_batch_size, valid_batch_size, test_images_amount, categories) 
    return data.get_all_batches()


def calculate_steps_per_epoch(paths_explorer):
    train_steps = int(paths_explorer.get_amount_of_train_images()/train_batch_size)
    valid_steps = int(paths_explorer.get_amount_of_valid_images()/valid_batch_size)
    return (train_steps, valid_steps)


def make_model_structure():
    
    model_platform = Create_model()
    model_platform.add_convolutional_layers(2, input_shape)
    model_platform.flatten_model()
    model_platform.add_neural_network_layers(units, activation)
    model_platform.compile_model(learning_rate)
    return model_platform.get_model_structure()
   

def train(model_structure, train_batches, valid_batches, train_steps, valid_steps):
    performance = Fit_model(model_structure)
    performance.run(train_batches, valid_batches, train_steps, valid_steps, epochs) 
    performance.plot_history()
    

def test(model, test_batches):
    prediction = Classify(model, categories)
    prediction.test(test_batches)
    prediction.print_results()
    
    
def main():
    paths_explorer = get_paths() 
    train_path, valid_path, test_path = paths_explorer.get_all_data_repository_paths()
    test_images_amount = paths_explorer.get_amount_of_test_images()
    train_batches, valid_batches, test_batches = make_data(train_path, valid_path, test_path, test_images_amount)    
    train_steps, valid_steps = calculate_steps_per_epoch(paths_explorer) 
    
    model_structure = make_model_structure()
    
    train(model_structure, train_batches, valid_batches, train_steps, valid_steps)
    test(model_structure, test_batches)
   
    
main() 
