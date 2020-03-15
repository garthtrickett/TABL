#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Dat Tran (dat.tranthanh@tut.fi)
"""
import Models
import keras
import numpy as np


# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40, 10], [60, 10], [120, 5], [3, 1]]

# random data
# random data
example_x = np.random.rand(1000, 40, 10)
example_y = keras.utils.to_categorical(np.random.randint(0, 3, (1000,)), 3)


## PRODIGY AI HOCKUS POCKUS START
from pathlib import Path
import h5py

home = str(Path.home())

file_name = "model_name=two_model&WL=10&pt=1&sl=1&min_ret=0.00151&vbs=600&head=0&skip=0&fraction=1&vol_max=0.00150002&vol_min=0.00150001&filter_type=none&cm_vol_mod=0&sample_weights=on&frac_diff=off&prices_type=orderbook&ntb=True&tslbc=True.h5"
path = home + "/ProdigyAI/data/preprocessed/" + file_name
h5f = h5py.File(path, "r")
X = h5f["X"][:]
y = h5f["y"][:]
h5f.close()
## PRODIGY AI HOCKUS POCKUS END


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

# X = X[0:200]
# y = y[0:200]

## Simple Training, Validation and Test splits
X_train = X[0 : round(0.78 * len(X))]
y_train = y[0 : round(0.78 * len(y))]

X_val = X_train[round(0.8 * len(X_train)) :]
y_val = y_train[round(0.8 * len(y_train)) :]

X_train = X_train[0 : round(0.78 * len(X_train))]
y_train = y_train[0 : round(0.78 * len(y_train))]


X_test = X[round(0.8 * len(X)) :]
y_test = y[round(0.8 * len(y)) :]


# training
# model.fit(x, y, batch_size=256, epochs=10000, class_weight=class_weight)
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=256,
    epochs=5000,
    shuffle=False,
)  # no class weight


score = model.evaluate(x=X_test, y=y_test, batch_size=256)

print(score)
