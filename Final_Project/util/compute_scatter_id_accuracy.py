#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 15:29:23 2023

@author: kristencirincione
"""

def compute_scatter_id_accuracy(pred, true):
    correct = 0
    points_used = []
    
    for xy_dict in true[0]:
        x = xy_dict['x']
        y = xy_dict['y']
        for point in pred:
            if abs(point[0] - x) <= 3 and abs(point[1] - y) <= 3 \
                and point not in points_used:
                correct += 1
                points_used.append(point)
                break
            
    return correct, len(true[0])


def compute_scatter_mean_percent_error(true_preds, true, label_preds, labels):
    x_error = 0
    y_error = 0
    nx = 0
    ny = 0
    count = 0
    points_used = []
    
    for xy_dict in true[0]:
        x = xy_dict['x']
        y = xy_dict['y']
        for point in true_preds:
            if abs(point[0] - x) <= 3 and abs(point[1] - y) <= 3 \
                and point not in points_used:
                
                points_used.append(point)
                
                error = (abs(label_preds[point][0] - labels[count]['x'])) \
                    / labels[count]['x']
                x_error += error
                nx += 1
                
                error = (abs(label_preds[point][1] - labels[count]['y'])) \
                    / labels[count]['y']
                y_error += error
                ny += 1
                
                break
                
        count += 1
    
    if nx > 0:
        mean_x_error = (x_error / nx) * 100
    else:
        mean_x_error = None
        
    if ny > 0:
        mean_y_error = (y_error / ny) * 100
    else:
        mean_y_error = None
    
    return mean_x_error, mean_y_error