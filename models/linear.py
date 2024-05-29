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

class Linear(Model):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice):
        super(Linear, self).__init__()
        self.target_slice = target_slice
        self.rev_norm = RevNorm(axis=-2)
        self.output_layer = layers.Dense(pred_len)

    def call(self, inputs, training=False):
        x = self.rev_norm(inputs, 'norm')
        
        if self.target_slice:
            x = x[:, :, self.target_slice]

        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.output_layer(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        outputs = self.rev_norm(x, 'denorm', self.target_slice)
        return outputs