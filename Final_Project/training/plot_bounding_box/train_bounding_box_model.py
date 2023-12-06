#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 12:37:09 2023

@author: kristencirincione
"""

from util.Data_Sources import train_images, min_width, min_height
from util.convert_to_grayscale import convert_to_grayscale
from training.plot_bounding_box.build_bb_model import build_bb_model
import numpy as np


img_resized = [img.resize_image(min_width, min_height) for img in train_images]
img_gray = [convert_to_grayscale(img) for img in img_resized]

Y_bb_temp= [img.plot_bb for img in train_images]
Y_bb = [np.array([y['x0'], y['y0'], y['width'], y['height']]) for y in Y_bb_temp]
X_img_ratio = [img.image_resized_ratio for img in train_images]
Y_bb_resized = [np.array(
    [y['x0']*X_img_ratio[i][1],
     y['y0']*X_img_ratio[i][0],
     y['width']*X_img_ratio[i][1], 
     y['height']*X_img_ratio[i][0]]) for i, y in enumerate(Y_bb_temp)]

X = np.array(img_gray)
Y = np.array(Y_bb_resized)

split = int(len(X) * 0.8)

X_train, Y_train = X[:split], Y[:split]
X_val, Y_val = X[split:], Y[split:]

model = build_bb_model(input_shape=(min_width, min_height, 1))

history = model.fit(X_train, Y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, Y_val)
)

model.save('./training/plot_bounding_box/models/model_1.keras')