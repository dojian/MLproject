#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:01:57 2023

@author: kristencirincione
"""
import numpy as np

def scale_scatter_points(points, x_scale, y_scale):
    
    scaled_points = {}
    for point in points:
        if x_scale[0] != None:
            x = int(x_scale[0]) + ((point[0] - x_scale[1]) * x_scale[2])
        else:
            x = None
            
        if y_scale[0] != None:
            y = int(y_scale[0]) + ((y_scale[1] - point[1]) * y_scale[2])
        else:
            y = None
        
        scaled_points[point] = (x, y)
        
    return scaled_points


def determine_x_scale(text, left, width):
    num_ind = [i for i in range(len(text)) if text[i].isnumeric()]
    
    if len(num_ind) > 1:
        
        axis_labels_temp = [text[i] for i in range(len(text)) if i in num_ind]
        lefts_temp = [left[i] for i in range(len(left)) if i in num_ind]
        widths_temp = [width[i] for i in range(len(width)) if i in num_ind]
        
        axis_labels = []
        lefts = []
        widths = []
        
        for i in range(len(axis_labels_temp) - 1):
            if int(axis_labels_temp[i]) < int(axis_labels_temp[i+1]):
                axis_labels.append(axis_labels_temp[i])
                lefts.append(lefts_temp[i])
                widths.append(widths_temp[i])
            if i == len(axis_labels_temp) - 2:
                axis_labels.append(axis_labels_temp[i+1])
                lefts.append(lefts_temp[i+1])
                widths.append(widths_temp[i+1])
        
        if len(axis_labels) > 1:
            scale = []
            for i in range(len(axis_labels) - 1):
                dif_1 = float(axis_labels[i+1]) - float(axis_labels[i])
                dif_2 = lefts[i+1] - lefts[i]
                if dif_2 > 0:
                    scale.append(dif_1/dif_2)
                
            scale_mean = np.mean(scale)
            scale_std = np.std(scale)
                    
            scale_final = []
            axis_labels_final = [axis_labels[0]]
            lefts_final = [lefts[0]]
            widths_final = [widths[0]]
                    
            for i in range(len(scale)):
                if abs(scale[i] - scale_mean) <= 1.5*scale_std:
                    scale_final.append(scale[i])
                    axis_labels_final.append(axis_labels[i+1])
                    lefts_final.append(lefts[i+1])
                    widths_final.append(widths[i+1])
            
            if len(axis_labels_final) > 1:
                reference_label = axis_labels[0]
                reference_location = lefts[0] + (widths[0] / 2)
                return reference_label, reference_location, np.mean(scale)
    
    return None, None, None


def determine_y_scale(text, top, height):
    num_ind = [i for i in range(len(text)) if text[i].isnumeric()]
    
    if len(num_ind) > 1:
        
        axis_labels_temp = [text[i] for i in range(len(text)) if i in num_ind]
        tops_temp = [top[i] for i in range(len(top)) if i in num_ind]
        heights_temp = [height[i] for i in range(len(height)) if i in num_ind]
        
        axis_labels = []
        tops = []
        heights = []
        
        for i in range(len(axis_labels_temp) - 1):
            if int(axis_labels_temp[i]) > int(axis_labels_temp[i+1]):
                axis_labels.append(axis_labels_temp[i])
                tops.append(tops_temp[i])
                heights.append(heights_temp[i])
            if i == len(axis_labels_temp) - 2:
                axis_labels.append(axis_labels_temp[i+1])
                tops.append(tops_temp[i+1])
                heights.append(heights_temp[i+1])
        
        if len(axis_labels) > 1:
            scale = []
            for i in range(len(axis_labels) - 1):
                dif_1 = float(axis_labels[i]) - float(axis_labels[i+1])
                dif_2 = tops[i+1] - tops[i]
                if dif_2 > 0:
                    scale.append(dif_1/dif_2)
                
            scale_mean = np.mean(scale)
            scale_std = np.std(scale)
            
            scale_final = []
            axis_labels_final = [axis_labels[0]]
            tops_final = [tops[0]]
            heights_final = [heights[0]]
                    
            for i in range(len(scale)):
                if abs(scale[i] - scale_mean) <= 1.5*scale_std:
                    scale_final.append(scale[i])
                    axis_labels_final.append(axis_labels[i+1])
                    tops_final.append(tops[i+1])
                    heights_final.append(heights[i+1])
            
            if len(axis_labels_final) > 1:
                reference_label = axis_labels_final[0]
                reference_location = tops_final[0] + (heights_final[0] / 2)
                return reference_label, reference_location, np.mean(scale_final)
    
    return None, None, None