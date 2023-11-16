#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:34:37 2023

@author: kristencirincione
"""
import tensorflow as tf

def build_model(input_shape):
    
    model = tf.keras.Sequential()

    # add first convolution layer to the model
    model.add(tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        name='conv_1',
        activation='relu'))
    
    
    # add a max pooling layer with pool size (2,2) and strides of 2
    # (this will reduce the spatial dimensions by half)
    model.add(tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='pool_1'))
    
    
    # add second convolutional layer
    # model.add(tf.keras.layers.Conv2D(
    #     filters=64,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     padding='same',
    #     name='conv_2',
    #     activation='relu'))
    
    # # add second max pooling layer with pool size (2,2) and strides of 2
    # # (this will further reduce the spatial dimensions by half)
    # model.add(tf.keras.layers.MaxPool2D(
    #     pool_size=(2, 2), name='pool_2')
    # )
    
    
    # add a fully connected layer (need to flatten the output of the previous layers first)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        units=16,
        name='fc_1',
        activation='relu'))
    
    # add dropout layer
    # model.add(tf.keras.layers.Dropout(
    #     rate=0.5))
    
    # add the last fully connected layer
    # this last layer sets the activation function to "None" in order to output the logits
    # note that passing activation = "sigmoid" will return class memembership probabilities but
    # in TensorFlow logits are prefered for numerical stability
    # set units=1 to get a single output unit (remember it's a binary classification problem)
    model.add(tf.keras.layers.Dense(
        units=5,
        name='fc_2',
        activation=None))
    
    model.build(input_shape=input_shape)
    
    return model