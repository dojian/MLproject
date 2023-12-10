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
from util.detect_scatter_points import detect_scatter_points
from util.determine_scale import determine_x_scale, determine_y_scale, scale_scatter_points
from util.compute_scatter_id_accuracy import compute_scatter_id_accuracy, compute_scatter_mean_percent_error


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
    # fig2, ax2 = plt.subplots()
    # ax2.imshow(img.image)
    # fig, ax = plt.subplots()
    # ax.imshow(img.image)
    # for point in points:
    #     ax.scatter(point[0], point[1], color='red')
    
good_preds = 0
total_preds = 0
count = 0
for img in scatter_images:
    good, total = compute_scatter_id_accuracy(scatter_points[count],
                                              scatter_images[count].scatter_points)
    good_preds += good
    total_preds += total
    count += 1
    
print('Scatter Plot Point ID Accuracy:', (good_preds/total_preds) * 100)
    
x_axis_images = [img.image[int(img.pred_plot_bb[1] + img.pred_plot_bb[3]):, :] 
                 for img in scatter_images]

custom_config = r'--oem 3 --psm 7'
x_axis_values = [pytesseract.image_to_data(img, 
                 output_type=Output.DICT, config=custom_config) for img in
                 x_axis_images]

x_axis_scale = []
for x in x_axis_values:
    ref, ref_loc, scale = determine_x_scale(x['text'], x['left'], x['width'])
    x_axis_scale.append((ref, ref_loc, scale))


y_axis_images = [img.image[:, :int(img.pred_plot_bb[0])] 
                 for img in scatter_images]

custom_config = r'--oem 3 --psm 6'
y_axis_values = [pytesseract.image_to_data(img, 
                 output_type=Output.DICT, config=custom_config) for img in
                 y_axis_images]

y_axis_scale = []
for y in y_axis_values:
    ref, ref_loc, scale = determine_y_scale(y['text'], y['top'], y['height'])
    y_axis_scale.append((ref, ref_loc, scale))
    
image_scale_id_count = 0
scatter_points_with_scale = []
x_axis_scale_final = []
y_axis_scale_final = []
scatter_images_final = []
for i in range(len(x_axis_scale)):
    if x_axis_scale[i][0] != None and y_axis_scale[i][0] != None:
        image_scale_id_count += 1
        scatter_points_with_scale.append(scatter_points[i])
        x_axis_scale_final.append(x_axis_scale[i])
        y_axis_scale_final.append(y_axis_scale[i])
        scatter_images_final.append(scatter_images[i])
 
mean_percent_x_error = []
mean_percent_y_error = []
for i in range(len(scatter_points_with_scale)):
    scaled_points = scale_scatter_points(scatter_points_with_scale[i], 
                                         x_axis_scale_final[i], 
                                         y_axis_scale_final[i])
    x_error, y_error = compute_scatter_mean_percent_error(
        scatter_points_with_scale[i], scatter_images_final[i].scatter_points, 
        scaled_points, scatter_images_final[i].plot_data_series)
    
    mean_percent_x_error.append(x_error)
    mean_percent_y_error.append(y_error)
