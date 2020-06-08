# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:57:57 2020

@author: omer
"""

import os

class Path_Detector: 

    def __init__(self, categories):
        self.train_path = None
        self.valid_path = None
        self.test_path = None
        self.categories = categories
        
    def set_train_path(self, given_path):
        if self.check_primary_folder(given_path):
            self.train_path = given_path
    def set_valid_path(self, given_path):
        if self.check_primary_folder(given_path):
            self.valid_path = given_path
    def set_test_path(self, given_path):
        if self.check_primary_folder(given_path):
            self.test_path = given_path
    
    def get_train_path(self):
        return self.train_path
    def get_valid_path(self):
        return self.valid_path
    def get_test_path(self):
        return self.test_path
    def get_all_data_repository_paths(self):
        return (self.train_path, self.valid_path, self.test_path)
        
    
    def receive_data_repository_paths(self):
        self.train_path = self.qualify_path("train")
        self.valid_path = self.qualify_path("validation")
        self.test_path = self.qualify_path("test")
        
    def qualify_path(self, database_role):
        
        path = input("give the path to {} dataset:\n".format(database_role))
        while not self.check_primary_folder(path):
            path = input("give the path to {} dataset:\n".format(database_role))
        return path
    
    def check_primary_folder(self, path):
        if not os.path.isdir(path):
            print ("Unqualified Path Message: The path needs to direct a folder")
            return False
        if not path.isascii():
            print ("Unqualified Path Message: The path needs to be written only in English")
            return False
        if self.is_empty(path):
            print ("Unqualified Path Message: The primary folder needs to include sub folders")
            return False
        if path == self.train_path or path == self.valid_path or path == self.test_path:
            print ("Unqualified Path Message: The path is already being used as a dataset")
            return False
        if not self.contain_right_labels(os.listdir(path)):
            print ("Unqualified Path Message: The primary folder needs to include sub folders that named after the labels which the model fits")
            print ("Note: the sub folders that named after the labels which the model fits, need to include pictures related to their label")
            return False

        return self.check_sub_folders(path)
                
    def check_sub_folders(self, path):
        for folder_name in self.categories:
            folder_path = os.path.join(path, folder_name)
            if not os.path.isdir(folder_path):
                print ("Unqualified Path Message: In the primary folder needs to have a folder that is named after {}".format(folder_name))
                return False
            if self.is_empty(folder_path):
                print ("Unqualified Path Message: At least one of the sub folders, that is named after the model's titles, is empty. ", end ="") 
                print("Every sub folder, that is named after the model's titles, needs to include images in jpg kind")
                return False
            if self.unqualified_image(folder_path):
                print ("Unqualified Path Message: All the files in the sub folders,", end="") 
                print ("that is named after the model's titles, must be images in jpg kind")
                return False
        return True
    
    def unqualified_image(self, folder_path):
         for file_name in os.listdir(folder_path):
             if not file_name.endswith(".jpg"):
                 return True
         return False
        
    def is_empty(self, path):
        return len(os.listdir(path)) == 0
    

    def numbering_images_within_primary_folder(self, path):
        for folder_name in os.listdir(path):
            folder_path = os.path.join(path, folder_name)
            if os.path.isdir(folder_path):
                self.numbering_images(folder_name, folder_path)
            
    def numbering_images(self, name, folder_path):
        for counter, file_name in enumerate(os.listdir(folder_path)):
            src = os.path.join( folder_path, file_name )
            new_name = name + str(counter+1) +".jpg"
            dst = os.path.join(folder_path ,new_name)
            os.rename(src, dst)
    
            
    def get_amount_of_train_images(self):
        return self.counting_images_within_primary_folder(self.train_path)
    def get_amount_of_valid_images(self):
        return self.counting_images_within_primary_folder(self.valid_path)
    def get_amount_of_test_images(self):
        return self.counting_images_within_primary_folder(self.test_path)
            
    def counting_images_within_primary_folder(self, path):
        summit = 0
        for folder_name in self.categories:
            folder_path = os.path.join(path, folder_name)
            summit += len(os.listdir(folder_path))
        return summit
            
    def contain_right_labels(self, arr):
        for label in self.categories:
            if label not in arr:
                return False
        return True
        
        
        
        
        
        
        