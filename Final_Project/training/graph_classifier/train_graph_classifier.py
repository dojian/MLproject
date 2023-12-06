#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:37:09 2023

@author: kristencirincione
"""

from util.Data_Sources import train_images, min_width, min_height
from util.convert_to_grayscale import convert_to_grayscale
import numpy as np
from training.graph_classifier.build_model import build_model
import tensorflow as tf


img_resized = [img.resize_image(min_width, min_height) for img in train_images]
img_gray = [convert_to_grayscale(img) for img in img_resized]

plot_type_map = {'scatter':0, 'line':1, 'dot':2, 'vertical_bar':3}
Y_plot_type = [img.plot_type for img in train_images]
Y_plot_type_mapped = [plot_type_map[plot_type] for plot_type in Y_plot_type]

X = np.array(img_gray)
Y = np.array(Y_plot_type_mapped)

split = int(len(X) * 0.8)

X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

model = build_model(input_shape=(None, min_width, min_height, 1))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, Y_val)
)

model.save('./training/graph_classifier/models/model_4.keras')