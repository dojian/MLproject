#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:31:10 2023

@author: kristencirincione
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:51:05 2023

@author: kristencirincione
"""
from PIL import Image, ImageDraw
from util.edge_detection import canny_edge_detector
import numpy as np
import cv2

def get_center_point(cluster):
    #cluster = flatten_2D_list(cluster)
    cluster = remove_outliers(cluster)
    
    x_min = cluster[0][0]
    x_max = cluster[0][0]
    y_min = cluster[0][1]
    y_max = cluster[0][1]
    
    for point in cluster:
        if point[0] < x_min:
            x_min = point[0]
        elif point[0] > x_max:
            x_max = point[0]
            
        if point[1] < y_min:
            y_min = point[1]
        elif point[1] > y_max:
            y_max = point[1]
                
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    return center
    

def remove_outliers(lst):
    #y = [x[1] for x in lsts]
    y_mean = np.mean(lst)
    y_std = np.std(lst)
            
    keep = []
    for point in lst:
        if abs(point[1] - y_mean) <= 2 * y_std:
            keep.append(point)
            
    return keep


def retrieve_clusters(lst):
    clusters = []
    new_lst = lst
    count = 0
    
    while len(new_lst) > 0 and count <= 50:
        out, new_lst = detect_cluster(new_lst)
        count += 1
        if len(out) > 0:
            clusters.append(out)
            
    return clusters


def detect_cluster(lst):
    lst = sorted(lst, key=lambda x: (x[0], x[1]))
    
    current_point = lst[0]
    x = current_point[0]
    y = current_point[1]
    cluster = [current_point]
    
    for point in lst[1:]:
        if point[0] <= x+1 and abs(point[1] - y) <= 20 :
            cluster.append(point)
            if x != point[0]:
                x = point[0]
    
    new_lst = [x for x in lst if x not in cluster]
    return cluster, new_lst


def remove_horizontal_lines(edges):
    groups = []
    group = []

    edges_sorted = sorted(edges, key=lambda x: (x[1], x[0]))
    y = edges_sorted[0][0]
    
    for edge in edges_sorted:
        if edge[1] == y:
            group.append(edge)
        else:
            y = edge[1]
            if len(group) > 0:
                groups.append(group)
            group = [edge]
            
    if len(group) > 0:
        groups.append(group)
        
    clusters = []
    for lst in groups:
        consecutive_count = 0
        x = lst[0][0]
        horizontal_line = False
        for point in lst:
            if consecutive_count >= 8:
                horizontal_line = True
                break
            if point[0] == x + 1:
                consecutive_count += 1
                x = point[0]
            else:
                consecutive_count = 0
                x = point[0]
        if not horizontal_line:
            clusters.append(lst)
            
    return flatten_2D_list(clusters)
    

def remove_vertical_lines(edges):
    groups = []
    group = []

    edges_sorted = sorted(edges, key=lambda x: (x[0], x[1]))
    x = edges_sorted[0][0]
    
    for edge in edges_sorted:
        if edge[0] == x:
            group.append(edge)
        else:
            x = edge[0]
            if len(group) > 0:
                groups.append(group)
            group = [edge]
            
    if len(group) > 0:
        groups.append(group)
    
    clusters = []
    for lst in groups:
        consecutive_count = 0
        y = lst[0][1]
        vertical_line = False
        for point in lst:
            if consecutive_count >= 8:
                vertical_line = True
                break
            if point[1] == y + 1:
                consecutive_count += 1
                y = point[1]
            else:
                consecutive_count = 0
                y = point[1]
        if not vertical_line:
            clusters.append(lst)
            
    return flatten_2D_list(clusters)


def flatten_2D_list(lst):
    flattened_lst = []
    
    for entries in lst:
        for entry in entries:
            flattened_lst.append(entry)
    return flattened_lst
    

def detect_scatter_points(img):
    
    x1 = int(img.pred_plot_bb[0])
    x2 = int(img.pred_plot_bb[0]) + int(img.pred_plot_bb[2])
    y1 = int(img.pred_plot_bb[1])
    y2 = int(img.pred_plot_bb[1]) + int(img.pred_plot_bb[3])
    
    image = Image.fromarray(img.image[y1:y2, x1:x2])
    edges = canny_edge_detector(image)

    edges_no_vertical = remove_vertical_lines(edges)
    edges_no_horizontal = remove_horizontal_lines(edges_no_vertical)
    edges_clustered = retrieve_clusters(edges_no_horizontal)
    edges_clustered = [edge for edge in edges_clustered if len(edge) >= 10]
    
    points = []
    for cluster in edges_clustered:
        center = get_center_point(cluster)
        center_adjusted = (center[0] + x1, center[1] + y1)
        points.append(center_adjusted)
    
    return points

    