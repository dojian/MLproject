#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 18:07:53 2023

@author: kristencirincione
"""
from Data_Loader import Data_Loader
from Image_Processor import Image_Processor
from build_model import build_model
import numpy as np
import tensorflow as tf
import pickle

data_file_path = './data/benetech-making-graphs-accessible/train/images'
label_file_path = './data/benetech-making-graphs-accessible/train/annotations'

n_images = 70000

# Load in raw data
X_raw, Y_raw = Data_Loader(data_file_path, 
                           label_file_path, n_images).load_image_data()

with open('X_raw.pickle', 'wb') as handle:
    pickle.dump(X_raw, handle)
    
with open('Y_raw.pickle', 'wb') as handle:
    pickle.dump(Y_raw, handle)

# X_processed = Image_Processor(X_raw)

# X_resized = X_processed.resized_images

# X_gray = X_resized / 255.0

# Y_plot_type = [y['chart-type'] for y in Y_raw]

# plot_type_mapping = {'scatter':0, 'line':1, 'dot':2, 'vertical_bar':3, 
#                       'horizontal_bar':4}
# Y_plot_type = [plot_type_mapping[plot_type] for plot_type in Y_plot_type]
# Y_plot_type = np.array(Y_plot_type)

# shuffle = tf.random.shuffle(tf.range(tf.shape(X_gray)[0], dtype=tf.int32))
# X_all = tf.gather(X_gray, shuffle)
# y_all = tf.gather(Y_plot_type, shuffle)

# split = int(len(X_raw) * 0.8)

# X_train, Y_train = X_all[:split], y_all[:split]
# X_val, Y_val = X_all[split:], y_all[split:]

# model = build_model(
#     input_shape=(None, X_processed.min_width, X_processed.min_height, 3))

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(X_train, Y_train,
#                     epochs=10,
#                     validation_data=(X_val, Y_val)
# )

#model.save('model_1.keras')
#model = keras.models.load_model('path/to/location.keras')