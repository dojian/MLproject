#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:25:05 2023

@author: kristencirincione
"""
import numpy as np

def get_smallest_image_dimensions(image_list):
    
        min_width = image_list[0].image_size[0]
        min_height = image_list[0].image_size[1]
        
        for img in image_list[1:]:
            
            if img.image_size[0] < min_width:
                min_width = img.image_size[0]
                
            if img.image_size[1] < min_height:
                min_height = img.image_size[1]
                
        return min_width, min_height