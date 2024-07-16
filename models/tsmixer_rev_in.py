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

"""Implementation of TSMixer with Reversible Instance Normalization."""
import numpy as np
from models.rev_in import RevNorm
from models.tsmixer import ResBlock
from tensorflow.keras import layers
import tensorflow as tf

class TSMixer(tf.keras.Model):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice, proposed, centers, fourier):
        super(TSMixer, self).__init__()
        self.pred_len = pred_len
        self.target_slice = target_slice
        self.proposed = proposed
        self.centers = centers
        self.fourier = fourier
        self.input_len = input_shape[0]
        self.input_dim = input_shape[1]  # 入力特徴量の次元数
        if proposed != 'base':
            self.supplement = Supplement(self.input_len, self.input_dim, centers, fourier)
        self.blocks = [ResBlock(norm_type, activation, dropout, ff_dim, self.input_len, self.input_dim) for _ in range(n_block)]
        self.output_layer = layers.Dense(pred_len)
        self.rev_norm = RevNorm(axis=-2)

    def call(self, inputs, training=False):
        x = self.rev_norm(inputs, 'norm')
        
        if self.proposed == 'base':
            supplement = None
        else:
            supplement = self.supplement(inputs, self.proposed, self.centers, self.fourier)
        
        for block in self.blocks:
            x = block(x, supplement)

        if self.target_slice:
            x = x[:, :, self.target_slice]
        x = tf.transpose(x, perm=[0, 2, 1])
        outputs = self.output_layer(x)
        outputs = tf.transpose(outputs, perm=[0, 2, 1])
        outputs = self.rev_norm(outputs, 'denorm', self.target_slice)
        return outputs
        
class Supplement(layers.Layer):
    def __init__(self, input_len, input_dim, centers, fourier):
        super(Supplement, self).__init__()
        self.length = input_len
        self.feature = input_dim
        self.centers = centers
        self.fourier_features = fourier
        self.num_lines = 5

        if fourier is not None:
            self.fourier_params = self._initialize_fourier_parameters()

    def _initialize_parameter(self, initial_values, name):
        return self.add_weight(
            name=name,
            shape=(len(initial_values),),
            initializer=tf.constant_initializer(np.array(initial_values)),
            trainable=True
        )

    def _initialize_fourier_parameters(self):
        params = {}
        for feature_name, components in self.fourier_features.items():
            for i, component in enumerate(components):
                for param_name in ['k', 'a', 'b']:
                    param_value = component[param_name]
                    key = f'{feature_name}_{i}_{param_name}'
                    params[key] = self.add_weight(
                        name=key,
                        shape=(),
                        initializer=tf.constant_initializer(param_value),
                        trainable=True
                    )
        return params

    def generate_fourier_index(self, batch_size):
        fourier_index = []
        for feature_name, components in self.fourier_features.items():
            for i, _ in enumerate(components):
                k = self.fourier_params[f'{feature_name}_{i}_k']
                a = self.fourier_params[f'{feature_name}_{i}_a']
                b = self.fourier_params[f'{feature_name}_{i}_b']
                radians = tf.linspace(0.0, 2 * np.pi, self.length)
                wave = a * tf.cos(k * radians) + b * tf.sin(k * radians)
                fourier_index.append(tf.reshape(wave, (1, self.length, 1)))

        fourier_index = tf.concat(fourier_index, axis=-1)
        return tf.tile(fourier_index, [batch_size, 1, 1])

    def calculate_similarity(self, inputs):
        return tf.matmul(inputs, self.centers, transpose_b=True)

    def call(self, inputs, supplement, centers, fourier):
        batch_size = tf.shape(inputs)[0]
        if supplement == "FAM":
            return self.generate_fourier_index(batch_size)

        elif supplement == "EPAM":
            return tf.matmul(inputs, self.centers, transpose_b=True)

        else:
            return None