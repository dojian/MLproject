#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:10:01 2023

@author: kristencirincione
"""

from util.Data_Loader import Data_Loader
from util.get_smallest_image_dimensions import get_smallest_image_dimensions

train_data_file_path = '../Final_Project/data/data_all/train/images'
train_label_file_path = '../Final_Project/data/data_all/train/annotations'

test_data_file_path = '../Final_Project/data/data_all/test/images'
test_label_file_path = '../Final_Project/data/data_all/test/annotations'

train_images = Data_Loader(train_data_file_path, 
                     train_label_file_path, 8000).load_image_data()

min_width, min_height = get_smallest_image_dimensions(train_images)

test_images = Data_Loader(test_data_file_path, 
                     test_label_file_path, 8).load_image_data()