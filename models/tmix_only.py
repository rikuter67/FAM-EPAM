# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer."""
import numpy as np
from models.rev_in import RevNorm
import tensorflow as tf
import pdb
from tensorflow.keras import layers, Model

class TMix_Only(Model):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice):
        super(TMix_Only, self).__init__()
        self.pred_len = pred_len
        self.target_slice = target_slice
        self.rev_norm = RevNorm(axis=-2)
        self.blocks = [ResBlockT(norm_type, activation, dropout, input_shape) for _ in range(n_block)]
        self.output_layer = layers.Dense(pred_len)

    def call(self, inputs, training=False):
        x = self.rev_norm(inputs, 'norm')
        for block in self.blocks:
            x = block(x, training=training)
        
        if self.target_slice:
            x = x[:, :, self.target_slice]

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.output_layer(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        outputs = self.rev_norm(x, 'denorm', self.target_slice)
        return outputs



class ResBlockT(layers.Layer):
    def __init__(self, norm_type, activation, dropout, input_shape):
        super(ResBlockT, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        self.dropout_rate = dropout
        self.norm = layers.LayerNormalization(axis=[-2, -1]) if norm_type == 'L' else layers.BatchNormalization(axis=[-2, -1])
        self.dense = layers.Dense(input_shape[0], activation=activation)
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.dense(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.dropout_layer(x, training=training)
        return x + inputs