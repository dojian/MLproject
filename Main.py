#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:07:53 2023

@author: kristencirincione
"""
from Data_Loader import Data_Loader
from Image_Processor import Image_Processor
from build_model import build_model
from build_bb_model import build_bb_model
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras

data_file_path = './data_reduced/train/images'
label_file_path = './data_reduced/train/annotations'

n_images = 8000

# Load in raw data
X_raw, Y_raw = Data_Loader(data_file_path, 
                           label_file_path, n_images).load_image_data()

X_processed = Image_Processor(X_raw)

X_resized = X_processed.resized_images

X_gray = X_resized / 255.0

Y_bb_temp= [y['plot-bb'] for y in Y_raw]

Y_bb = [np.array([y['x0'], y['y0'], y['width'], y['height']]) for y in Y_bb_temp]

X_img_ratio = X_processed.img_ratio

Y_bb_resized = [np.array([y['x0']*X_img_ratio[i][1],
                          y['y0']*X_img_ratio[i][0],
                          y['width']*X_img_ratio[i][1], 
                          y['height']*X_img_ratio[i][0]]) for i, y in enumerate(Y_bb_temp)]

#Y_plot_type = [y['chart-type'] for y in Y_raw]

#plot_type_mapping = {'scatter':0, 'line':1, 'dot':2, 'vertical_bar':3, 
#                      'horizontal_bar':4}
#Y_plot_type = [plot_type_mapping[plot_type] for plot_type in Y_plot_type]
#Y_plot_type = np.array(Y_plot_type)

#shuffle = tf.random.shuffle(tf.range(tf.shape(X_gray)[0], dtype=tf.int32))
#X_all = tf.gather(X_gray, shuffle)
#y_all = tf.gather(Y_plot_type, shuffle)
#y_all = tf.gather(Y_bb_resized, shuffle)

#split = int(len(X_raw) * 0.8)

#X_train, Y_train = X_all[:split], y_all[:split]
#X_val, Y_val = X_all[split:], y_all[split:]

model = keras.models.load_model('model_bb_13_best.keras')

#model = build_model(
#    input_shape=(None, X_processed.min_width, X_processed.min_height, 3))

#model.compile(optimizer=tf.keras.optimizers.Adam(),
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])

# model = build_bb_model(input_shape=(X_processed.min_width, X_processed.min_height, 3))

# history = model.fit(X_train, Y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(X_val, Y_val)
# )

#model.save('model_1.keras')
#model = keras.models.load_model('path/to/location.keras')