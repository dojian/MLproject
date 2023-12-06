#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:41:26 2023

@author: kristencirincione
"""
import cv2
import numpy as np

def convert_to_grayscale(img):
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.00