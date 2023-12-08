#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:27:45 2023

@author: kristencirincione
"""

from util.Data_Sources import test_images
from util.convert_to_grayscale import convert_to_grayscale
import numpy as np
import keras
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
from util.detect_scatter_points_v2 import detect_scatter_points

def determine_scale(img, vals, height, left):
    num_ind = [i for i in range(len(vals)) if vals[i].isnumeric()]
    axis_labels = [vals[i] for i in range(len(vals)) if i in num_ind]
    heights = [height[i] for i in range(len(height)) if i in num_ind]
    lefts = [left[i] for i in range(len(left)) if i in num_ind]
    
    difs = []
    for i in range(len(axis_labels) - 1):
        dif_1 = float(axis_labels[i+1]) - float(axis_labels[i])
        dif_2 = lefts[i+1] - lefts[i]
        difs.append((dif_1, dif_2))

img_resized = [img.resize_image(145, 318) for img in test_images]
img_gray = [convert_to_grayscale(img) for img in img_resized]

# Run plot classifier on test images
plot_type_map = {'scatter':0, 'line':1, 'dot':2, 'vertical_bar':3}
Y_plot_type = [img.plot_type for img in test_images]
Y_plot_type_mapped = [plot_type_map[plot_type] for plot_type in Y_plot_type]

X_plot_type = np.array(img_gray)
Y_plot_type = np.array(Y_plot_type_mapped)

graph_classifier = keras.models.load_model(
    './training/graph_classifier/models/model_3.keras'
)

plot_type_pred_logits = graph_classifier.predict(X_plot_type)
plot_type_preds = np.argmax(plot_type_pred_logits, axis=-1)

# Run bounding box model on test images
Y_bb_temp= [img.plot_bb for img in test_images]
Y_bb = [np.array([y['x0'], y['y0'], y['width'], y['height']]) for y in Y_bb_temp]
X_img_ratio = [img.image_resized_ratio for img in test_images]
Y_bb_resized = [np.array(
    [y['x0']*X_img_ratio[i][1],
     y['y0']*X_img_ratio[i][0],
     y['width']*X_img_ratio[i][1], 
     y['height']*X_img_ratio[i][0]]) for i, y in enumerate(Y_bb_temp)]

X_bb = np.array(img_gray)
Y_bb = np.array(Y_bb_resized)

bounding_box_model = keras.models.load_model(
    './training/plot_bounding_box/models/model_1.keras'
)

plot_bb_preds = bounding_box_model.predict(X_bb)

# Examine scatter plots
scatter_images = []

for i in range(len(Y_plot_type)):
    if Y_plot_type[i] == 0 and plot_type_preds[i] == 0:
        test_images[i].store_pred_plot_bb(plot_bb_preds[i])
        scatter_images.append(test_images[i])
        
scatter_points = []

for img in scatter_images:
    points = detect_scatter_points(img)
    scatter_points.append(points)
    fig2, ax2 = plt.subplots()
    ax2.imshow(img.image)
    fig, ax = plt.subplots()
    ax.imshow(img.image)
    for point in points:
        ax.scatter(point[0], point[1], color='red')
    

x_axis_images = [img.image[int(img.pred_plot_bb[1] + img.pred_plot_bb[3]):, :] 
                 for img in scatter_images]

custom_config = r'--oem 3 --psm 7'
x_axis_values = [pytesseract.image_to_data(img, 
                 output_type=Output.DICT, config=custom_config) for img in
                 x_axis_images]

determine_scale(x_axis_images[0], x_axis_values[0]['text'], 
                x_axis_values[0]['height'], x_axis_values[0]['left'])

y_axis_images = [img.image[:, :int(img.pred_plot_bb[0])] 
                 for img in scatter_images]
