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

import tensorflow as tf
import pdb
from tensorflow.keras import layers

# 特徴Mixingのカスタマイズ
class FeatureMixingLayer(layers.Layer):
    def __init__(self, length):
        super(FeatureMixingLayer, self).__init__()
        self.length = length
    
    def call(self, inputs):
        # get batch_size
        batch_size = tf.shape(inputs)[0]

        # add index
        index = tf.linspace(1.0 / self.length, 1.0, self.length)
        index = tf.reshape(index, (1, self.length, 1))
        index = tf.tile(index, [batch_size, 1, 1])
        
        # calculate similarity
        mean_vector = tf.reduce_mean(inputs, axis=1, keepdims=True)
        similarity = tf.matmul(inputs, mean_vector, transpose_b=True)

        # concat index and similarity
        input_custom = tf.concat([inputs, similarity], axis=-1)

        return input_custom

def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer with positional encoding and mean vector inner product."""
    norm = (
        layers.LayerNormalization
        if norm_type == 'L'
        else layers.BatchNormalization
    )

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])
    x = layers.Dropout(dropout)(x)
    res = x + inputs


    # Feature Mixing Layerの適用
    feature_mixing = FeatureMixingLayer(res.shape[1])
    x = feature_mixing(res)

    # Feature Linear
    x = norm(axis=[-2, -1])(x)
    x = layers.Dense(ff_dim, activation=activation)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)

    return x + res


def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)

  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

  return tf.keras.Model(inputs, outputs)