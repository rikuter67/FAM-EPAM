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
import tensorflow.compat.v1 as tf
import numpy
from tensorflow.keras import layers

class ResBlock(layers.Layer):
    def __init__(self, norm_type, activation, dropout, ff_dim, input_len, input_dim):
        super(ResBlock, self).__init__()
        self.norm_type = norm_type
        self.activation = activation
        self.dropout_rate = dropout
        self.ff_dim = ff_dim
        self.input_len = input_len
        self.input_dim = input_dim

        if norm_type == 'L':
            self.norm1 = layers.LayerNormalization(axis=-1)
            self.norm2 = layers.LayerNormalization(axis=-1)
        else:
            self.norm1 = layers.BatchNormalization(axis=-1)
            self.norm2 = layers.BatchNormalization(axis=-1)
        # self.norm2 = layers.BatchNormalization() if norm_type == 'B' else layers.LayerNormalization()
        
        self.temporal_linear = layers.Dense(self.input_len, activation=activation, name=f"{self.name}_temporal_linear")
        self.feature_linear_1 = layers.Dense(self.ff_dim, activation=activation, name=f"{self.name}_feature_linear_1")
        self.feature_linear_2 = layers.Dense(self.input_dim, name=f"{self.name}_feature_linear_2")
        self.dropout1 = layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout1")
        self.dropout2 = layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout2")
        self.dropout3 = layers.Dropout(self.dropout_rate, name=f"{self.name}_dropout3")

    def get_sublayers(self):
        return {
            'temporal_linear': self.temporal_linear,
            'feature_linear_1': self.feature_linear_1,
            'feature_linear_2': self.feature_linear_2
        }

    def call(self, t_inputs, supplement):
        x = self.norm1(t_inputs)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.temporal_linear(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.dropout1(x)
        res = x + t_inputs

        if supplement != None :
            f_inputs = tf.concat([res, supplement], axis=-1)
        else :
            f_inputs = res

        x = self.norm2(f_inputs)
        x = self.feature_linear_1(x)
        x = self.dropout2(x)
        x = self.feature_linear_2(x)
        output = self.dropout3(x)

        return output + res