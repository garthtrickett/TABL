#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""

from third_party_libraries.TABL import Layers
import keras
import tensorflow as tf
# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def BL(template, dropout=0.1, regularizer=None, constraint=None):
    """
    Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    regularizer: keras regularizer object
    constraint: keras constraint object
    
    outputs
    ------
    keras model object
    """
    print(template)
    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        x = Layers.BL(template[k], regularizer, constraint)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    x = Layers.BL(template[-1], regularizer, constraint)(x)
    outputs = keras.layers.Activation("softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    opt = keras.optimizers.Adam(0.01)
    # opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    #     opt, "dynamic")
    model.compile(opt, "categorical_crossentropy", ["acc"])

    return model


def TABL(
    template,
    dropout=0.1,
    projection_regularizer=None,
    projection_constraint=None,
    attention_regularizer=None,
    attention_constraint=None,
):
    """
    Temporal Attention augmented Bilinear Layer network, refer to the paper https://arxiv.org/abs/1712.00975
    
    inputs
    ----
    template: a list of network dimensions including input and output, e.g., [[40,10], [120,5], [3,1]]
    dropout: dropout percentage
    projection_regularizer: keras regularizer object for projection matrices
    projection_constraint: keras constraint object for projection matrices
    attention_regularizer: keras regularizer object for attention matrices
    attention_constraint: keras constraint object for attention matrices
    
    outputs
    ------
    keras model object
    """

    inputs = keras.layers.Input(template[0])

    x = inputs
    for k in range(1, len(template) - 1):
        x = Layers.BL(template[k], projection_regularizer,
                      projection_constraint)(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Dropout(dropout)(x)

    x = Layers.TABL(
        template[-1],
        projection_regularizer,
        projection_constraint,
        attention_regularizer,
        attention_constraint,
    )(x)
    outputs = keras.layers.Activation("softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    opt = keras.optimizers.Adam(0.01)
    # opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(
    #     opt, "dynamic")
    model.compile(opt, "categorical_crossentropy", ["acc"])

    return model
