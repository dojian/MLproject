#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:03:12 2023

@author: kristencirincione
"""
import os
import cv2
import json


class Data_Loader():
    
    def __init__(self, data_file_path, label_file_path, n_images):
        
        self.data_file_path = data_file_path
        self.label_file_path = label_file_path
        self.n_images = n_images
    
    
    def load_image_data(self):
        
        X = []
        Y = []
        
        for file in os.listdir(self.data_file_path):
            
            file_path = os.path.join(self.data_file_path, file)
            
            if file.endswith('.jpg'):
                
                img_annotations = self.__load_annotations(file)
                
                if img_annotations is not None:
                    
                    Y.append(img_annotations)
                
                    img = cv2.imread(file_path)
                
                    X.append(img)
                
                if len(X) >= self.n_images :
                    
                    return X, Y
                        
        return X, Y
    
    
    def __load_annotations(self, image_file_name):
        
        file_name = image_file_name.split('.jpg')[0]
        
        json_file_name = file_name + '.json'
        
        json_file_path = os.path.join(self.label_file_path, json_file_name)
        
        if os.path.isfile(json_file_path):
            
            f = open(json_file_path)
            
            return json.load(f)
        
        else:
            return None
        