#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""
import Models
import keras
import numpy as np


# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [120, 5], [3, 1]]

# random data
# random data
example_x = np.random.rand(1000, 40, 10)
example_y = keras.utils.to_categorical(np.random.randint(0, 3, (1000,)), 3)


## PRODIGY AI HOCKUS POCKUS START
from pathlib import Path
import h5py

home = str(Path.home())

file_name = "model_name=two_model&WL=10&pt=1&sl=1&min_ret=0.0021&vbs=600&head=0&skip=0&fraction=1&vol_max=0.0022&vol_min=0.00210001&filter_type=none&cm_vol_mod=0&sample_weights=on&frac_diff=off&prices_type=orderbook&ntb=True&tslbc=True.h5"
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
x = h5f["X"][:]
y = h5f["y"][:]
h5f.close()

# get Bilinear model
projection_regularizer = None
projection_constraint = keras.constraints.max_norm(3.0, axis=0)
attention_regularizer = None
attention_constraint = keras.constraints.max_norm(5.0, axis=1)
dropout = 0.1


model = Models.TABL(
    template,
    dropout,
    projection_regularizer,
    projection_constraint,
    attention_regularizer,
    attention_constraint,
)
model.summary()

# create class weight
class_weight = {0: 1e6 / 300.0, 1: 1e6 / 400.0, 2: 1e6 / 300.0}


# training
# model.fit(x, y, batch_size=256, epochs=10000, class_weight=class_weight)
model.fit(x, y, batch_size=256, epochs=500)  # no class weight
