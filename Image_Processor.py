#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:03:25 2023

@author: kristencirincione
"""
import tensorflow as tf
import numpy as np


class Image_Processor():
    
    def __init__(self, images):
        self.images = images
        self.min_width, self.min_height = self.__find_smallest_image_width_and_height()
        self.resized_images = self.__resize_images()
    
    
    def __resize_images(self):
        
        X_resized = []
        
        for img in self.images:
            X_resized.append(tf.image.resize(img, 
                                             size=(self.min_width, self.min_height)))
            
        return np.array(X_resized)
    
    
    def __find_smallest_image_width_and_height(self):
        
        min_width = np.size(self.images[0], 0)
        min_height = np.size(self.images[0], 1)
        
        for img in self.images[1:]:
            
            if np.size(img, 0) < min_width:
                min_width = np.size(img, 0)
                
            if np.size(img, 1) < min_height:
                min_height = np.size(img, 1)
                
        return min_width, min_height


