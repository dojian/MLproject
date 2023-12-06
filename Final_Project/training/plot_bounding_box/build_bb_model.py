#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:01:20 2023

@author: kristencirincione
"""
import tensorflow as tf


def build_bb_model(input_shape):
    
    image = tf.keras.layers.Input(shape=input_shape, name='Image')
    
    conv_1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',
        name='conv_1',
        activation='relu')(image)
    
    pool = tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        name='pool_1')(conv_1)
    
    # conv_2 = tf.keras.layers.Conv2D(
    #     filters=32,
    #     kernel_size=(5, 5),
    #     strides=(1, 1),
    #     padding='same',
    #     data_format='channels_last',
    #     name='conv_2',
    #     activation='relu')(pool)
    
    # pool_2 = tf.keras.layers.MaxPool2D(
    #     pool_size=(2, 2),
    #     name='pool_2')(conv_2)
    
    flat = tf.keras.layers.Flatten()(pool)
    
    fc_1 = tf.keras.layers.Dense(
        units=16,
        name='fc_1',
        activation='relu')(flat)
    
    # add dropout layer
    #drop = tf.keras.layers.Dropout(rate=0.1)(fc_1)
    
    # fc_2 = tf.keras.layers.Dense(
    #     units=8,
    #     name='fc_2',
    #     activation='relu')(fc_1)
    
    out = tf.keras.layers.Dense(
        units=4,
        name='fc_3',
        activation=None)(fc_1)
    
    model = tf.keras.Model(inputs=image,
                         outputs=out, name='Bounding_Box')
    
    model.compile(
     optimizer=tf.keras.optimizers.Adam(),
     loss='mse',
     metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    return model