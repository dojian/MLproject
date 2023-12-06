#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:03:58 2023

@author: kristencirincione
"""
import os
import cv2
import json
from util.Plot_Image import Plot_Image


class Data_Loader():
    
    def __init__(self, data_file_path, label_file_path, n_images):
        
        self.data_file_path = data_file_path
        self.label_file_path = label_file_path
        self.n_images = n_images
    
    
    def load_image_data(self):
        
        X = []
        
        for file in os.listdir(self.data_file_path):
            
            file_path = os.path.join(self.data_file_path, file)
            
            if file.endswith('.jpg'):
                
                img_annotations, json_path = self.__load_annotations(file)
                
                if img_annotations is not None and \
                img_annotations['chart-type'] != 'horizontal_bar':
                
                    img = cv2.imread(file_path)
                
                    X.append(Plot_Image(img, file, 
                                        (img.shape[0], img.shape[1]), 
                                        img_annotations['chart-type'], 
                                        img_annotations['plot-bb'], 
                                        img_annotations['data-series']))
                
                if len(X) >= self.n_images :
                    
                    return X
                        
        return X
    
    
    def __load_annotations(self, image_file_name):
        
        file_name = image_file_name.split('.jpg')[0]
        
        json_file_name = file_name + '.json'
        
        json_file_path = os.path.join(self.label_file_path, json_file_name)
        
        if os.path.isfile(json_file_path):
            
            f = open(json_file_path)
            
            return json.load(f), json_file_path
        
        else:
            return None