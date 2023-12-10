#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 11:57:18 2023

@author: kristencirincione
"""
import copy
import tensorflow as tf
import numpy as np

class Plot_Image():
    
    def __init__(self, image, image_name, image_size, plot_type, plot_bb, 
                 plot_data_series, scatter_points):
        self.image = image
        self.image_name = image_name
        self.image_size = image_size
        self.plot_type = plot_type
        self.plot_bb = plot_bb
        self.plot_data_series = plot_data_series
        self.scatter_points = scatter_points
        
        
    def resize_image(self, width, height):
        
        img_resized = copy.deepcopy(self.image)
        
        image_resized = tf.image.resize(img_resized, 
                                             size=(width, height))
        
        self.image_resized_ratio = (width/self.image_size[0], 
                                    height/self.image_size[1])
        
        return np.array(image_resized,np.uint8)
    
    
    def store_pred_plot_bb(self, pred_bb_unscaled):
        
        self.pred_plot_bb = [
            pred_bb_unscaled[0] / self.image_resized_ratio[1],
            pred_bb_unscaled[1] / self.image_resized_ratio[0],
            pred_bb_unscaled[2] / self.image_resized_ratio[1],
            pred_bb_unscaled[3] / self.image_resized_ratio[0]
            ]
        