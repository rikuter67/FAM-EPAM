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

def generate_fourier_index(self, batch_size, fourier):
    fourier_index = []

    for feature_name, components in fourier.items():
        for component in components:
            k = component['k']
            a = component['a']
            b = component['b']
            radians = tf.linspace(0.0, 1, self.length)
            wave = a * tf.cos(2 * np.pi * k / self.length * radians) + b * tf.sin(2 * np.pi * k / self.length * radians)
            fourier_index.append(tf.reshape(wave, (1, self.length, 1)))

    fourier_index = tf.concat(fourier_index, axis=-1)  # すべてのフーリエ成分を結合
    return tf.tile(fourier_index, [batch_size, 1, 1])

def fourier(self, inputs, fourier):
    batch_size = tf.shape(inputs)[0]
    fourier_index = self.generate_fourier_index(batch_size, fourier)
    return tf.concat([inputs, fourier_index], axis=-1)

class FMix_Only(Model):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice, centers, fourier):
        super(FMix_Only, self).__init__()
        self.pred_len = pred_len
        self.target_slice = target_slice
        self.rev_norm = RevNorm(axis=-2)
        self.blocks = [ResBlockF(norm_type, activation, dropout, input_shape, ff_dim, centers, fourier) for _ in range(n_block)]
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



class ResBlockF(layers.Layer):
    def __init__(self, norm_type, activation, dropout, input_shape, ff_dim, centers, fourier):
        super(ResBlockF, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        self.dropout_rate = dropout
        self.ff_dim = ff_dim
        self.centers = centers
        self.fourier = fourier
        self.norm = layers.LayerNormalization(axis=[-2, -1]) if norm_type == 'L' else layers.BatchNormalization(axis=[-2, -1])
        self.feature_linear1 = layers.Dense(self.ff_dim, activation=activation)
        self.feature_linear2 = layers.Dense(input_shape[-1])
        self.dropout_layer = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x = tf.concat([inputs, tf.matmul(inputs, self.centers, transpose_b=True)], axis=-1)
        # x = fourier(input, self.fourier)
        x = self.norm(x)
        x = self.feature_linear1(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = self.feature_linear2(x)
        x = layers.Dropout(self.dropout_rate)(x)
        return x + inputs
