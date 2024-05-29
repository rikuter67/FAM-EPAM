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

"""Train and evaluate models for time series forecasting."""
import argparse
import glob
import logging
import os
import time

from data_loader import TSFDataLoader
from models.tsmixer_rev_in import TSMixer
from models.tmix_only import TMix_Only
from models.fmix_only import FMix_Only
from models.linear import Linear
import utils

import models
import numpy as np
import pandas as pd
import tensorflow as tf

import graph
from graph import PlotLayerOutputCallback
from graph import ResBlockIOCallback
import pdb
from sklearn.cluster import KMeans
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def parse_args():
  """Parse the arguments for experiment configuration."""

  parser = argparse.ArgumentParser(
      description='TSMixer for Time Series Forecasting'
  )

  # basic config
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument(
      '--model',
      type=str,
      default='tsmixer',
      help='model name, options: [tsmixer, tsmixer_rev_in]',
  )

  # data loader
  parser.add_argument(
      '--data',
      type=str,
      default='weather',
      choices=[
          'electricity',
          'exchange_rate',
          'national_illness',
          'traffic',
          'weather',
          'ETTm1',
          'ETTm2',
          'ETTh1',
          'ETTh2',
      ],
      help='data name',
  )
  parser.add_argument(
      '--feature_type',
      type=str,
      default='M',
      choices=['S', 'M', 'MS'],
      help=(
          'forecasting task, options:[M, S, MS]; M:multivariate predict'
          ' multivariate, S:univariate predict univariate, MS:multivariate'
          ' predict univariate'
      ),
  )
  parser.add_argument(
      '--target', type=str, default='OT', help='target feature in S or MS task'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoints/',
      help='location of model checkpoints',
  )
  parser.add_argument(
      '--delete_checkpoint',
      action='store_true',
      help='delete checkpoints after the experiment',
  )

  # forecasting task
  parser.add_argument(
      '--seq_len', type=int, default=512, help='input sequence length'
  )
  parser.add_argument(
      '--pred_len', type=int, default=96, help='prediction sequence length'
  )

  # model hyperparameter
  parser.add_argument(
      '--n_block',
      type=int,
      default=2,
      help='number of block for deep architecture',
  )
  parser.add_argument(
      '--ff_dim',
      type=int,
      default=2048,
      help='fully-connected feature dimension',
  )
  parser.add_argument(
      '--dropout', type=float, default=0.05, help='dropout rate'
  )
  parser.add_argument(
      '--norm_type',
      type=str,
      default='B',
      choices=['L', 'B'],
      help='LayerNorm or BatchNorm',
  )
  parser.add_argument(
      '--activation',
      type=str,
      default='relu',
      choices=['relu', 'gelu'],
      help='Activation function',
  )
  parser.add_argument(
      '--kernel_size', type=int, default=4, help='kernel size for CNN'
  )
  parser.add_argument(
      '--temporal_dim', type=int, default=16, help='temporal feature dimension'
  )
  parser.add_argument(
      '--hidden_dim', type=int, default=64, help='hidden feature dimension'
  )

  # optimization
  parser.add_argument(
      '--num_workers', type=int, default=10, help='data loader num workers'
  )
  parser.add_argument(
      '--train_epochs', type=int, default=100, help='train epochs'
  )
  parser.add_argument(
      '--batch_size', type=int, default=32, help='batch size of input data'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='optimizer learning rate',
  )
  parser.add_argument(
      '--patience', type=int, default=5, help='number of epochs to early stop'
  )

  # save results
  parser.add_argument(
      '--result_path', default='result.csv', help='path to save result'
  )

  parser.add_argument(
       '--training', default='train', choices=['train', 'test']
  )

  parser.add_argument(
      '--supplement',
      type=str,
      default='base',
      choices=['base', 'sequence', 'frequence', 'similarity', 'tmix', 'fmix', 'linear'],
      help='Proposed Method',
  )

  args = parser.parse_args()

  tf.random.set_seed(args.seed)
  np.random.seed()

  return args


def main():
  args = parse_args()
  if 'tsmixer' in args.model:
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
  elif args.model == 'tmix_only':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
  elif args.model == 'fmix_only':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
  elif args.model == 'linear':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
  elif args.model == 'full_linear':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}'
  elif args.model == 'cnn':
    exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_ks{args.kernel_size}'
  else:
    raise ValueError(f'Unknown model type: {args.model}')

  # load datasets
  data_loader = TSFDataLoader(
      args.data,
      args.batch_size,
      args.seq_len,
      args.pred_len,
      args.feature_type,
      args.target,
  )
  train_data = data_loader.get_train()
  val_data = data_loader.get_val()
  test_data = data_loader.get_test()

  # train model
  if 'tsmixer' in args.model:
    fourier, centers = utils.prepare_features(data_loader) if args.supplement != 'base' else ([], [])
    # print(fourier)
    model = TSMixer(
      input_shape=(args.seq_len, data_loader.n_feature),
      pred_len=args.pred_len,
      norm_type=args.norm_type,
      activation=args.activation,
      dropout=args.dropout,
      n_block=args.n_block,
      ff_dim=args.ff_dim,
      target_slice=data_loader.target_slice,
      proposed=args.supplement,
      centers=centers,
      fourier=fourier
    )
  elif args.model == 'tmix_only':
    model = TMix_Only(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
    )
  elif args.model == 'fmix_only':
    model = FMix_Only(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
        centers=data_loader.centers,
        fourier=fourier
    )
  elif args.model == 'linear':
    model = Linear(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        norm_type=args.norm_type,
        activation=args.activation,
        dropout=args.dropout,
        n_block=args.n_block,
        ff_dim=args.ff_dim,
        target_slice=data_loader.target_slice,
    )
  elif args.model == 'full_linear':
    model = models.full_linear.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
    )
  elif args.model == 'cnn':
    model = models.cnn.Model(
        n_channel=data_loader.n_feature,
        pred_len=args.pred_len,
        kernel_size=args.kernel_size,
    )
  else:
    raise ValueError(f'Model not supported: {args.model}')

  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
  model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
  checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_id}_best')
  if args.training == 'train':
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=args.patience
    )
    start_training_time = time.time()

    # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'temporal_linear', time_phase='before')
    # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'feature_linear1', time_phase='before')
    # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'feature_linear2', time_phase='before')


    history = model.fit(
        train_data,
        epochs=args.train_epochs,
        validation_data=val_data,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    end_training_time = time.time()
    elasped_training_time = end_training_time - start_training_time
    print(f'Training finished in {elasped_training_time} secconds')

  # evaluate best model
    best_epoch = np.argmin(history.history['val_loss'])


#   for layer in model.layers:
#     print(layer.summary())

# 重みのプロット
  # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'temporal_linear', time_phase='after')
  # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'feature_linear1', time_phase='after')
  # graph.plot_res_block_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len, layer_name = 'feature_linear2', time_phase='after')
  # graph.plot_last_dense_layer_weights(model, f'graph/weights/{args.supplement}', args.data, args.pred_len)

  
#   graph.history(history, save_path=f'graph/{args.supplement}/train_valid_{args.data}_{args.model}_{args.pred_len}_{args.supplement}.png')
  
  model.load_weights(checkpoint_path)
  graph.predictions_with_history(test_data, args.data, model, seq_len=args.seq_len, pred_len=args.pred_len, save_path=f'graph/{args.supplement}/{args.data}_{args.pred_len}/{args.supplement}')
  # layer_names_all = ['input_1', 'rev_norm', 'batch_normalization', 'tf.compat.v1.transpose', 'dense', 'tf.compat.v1.transpose_1', 'dropout', 'tf.__operators__.add', 'feature_mixing_layer', 'batch_normalization_1', 'dense_1', 'dropout_1', 'dense_2', 'dropout_2', 'tf.__operators__.add_1', 'batch_normalization_2', 'tf.compat.v1.transpose_2', 'dense_3', 'tf.compat.v1.transpose_3', 'dropout_3', 'tf.__operators__.add_2', 'feature_mixing_layer_1', 'batch_normalization_3', 'dense_4', 'dropout_4', 'dense_5', 'dropout_5', 'tf.__operators__.add_3', 'batch_normalization_4', 'tf.compat.v1.transpose_4', 'dense_6', 'tf.compat.v1.transpose_5', 'dropout_6', 'tf.__operators__.add_4', 'feature_mixing_layer_2', 'batch_normalization_5', 'dense_7', 'dropout_7', 'dense_8', 'dropout_8', 'tf.__operators__.add_5', 'batch_normalization_6', 'tf.compat.v1.transpose_6', 'dense_9', 'tf.compat.v1.transpose_7', 'dropout_9', 'tf.__operators__.add_6', 'feature_mixing_layer_3', 'batch_normalization_7', 'dense_10', 'dropout_10', 'dense_11', 'dropout_11', 'tf.__operators__.add_7', 'batch_normalization_8', 'tf.compat.v1.transpose_8', 'dense_12', 'tf.compat.v1.transpose_9', 'dropout_12', 'tf.__operators__.add_8', 'feature_mixing_layer_4', 'batch_normalization_9', 'dense_13', 'dropout_13', 'dense_14', 'dropout_14', 'tf.__operators__.add_9', 'batch_normalization_10', 'tf.compat.v1.transpose_10', 'dense_15', 'tf.compat.v1.transpose_11', 'dropout_15', 'tf.__operators__.add_10', 'feature_mixing_layer_5', 'batch_normalization_11', 'dense_16', 'dropout_16', 'dense_17', 'dropout_17', 'tf.__operators__.add_11', 'tf.__operators__.getitem', 'tf.compat.v1.transpose_12', 'dense_18', 'tf.compat.v1.transpose_13']
#   layer_names = ['input_1', 'batch_normalization']
#   layer_names.append(model.layers[-1].name)
#   for layer in model.layers:
#       if layer.name.startswith('tf.__operators'):
#           layer_names.append(layer.name)
#   plot_callback = PlotLayerOutputCallback(layer_names=layer_names, test_data=test_data)
#   test_result = model.evaluate(test_data, callbacks=[plot_callback])

  # ResBlockの名前を取得
  # res_block_names = [layer.name for layer in model.layers if 'res_block' in layer.name]

  # テストデータでモデルを評価し、入出力をプロット
  # res_block_io_callback = ResBlockIOCallback(model, test_data, res_block_names, f'graph/{args.supplement}/{args.data}_{args.pred_len}')

  test_result = model.evaluate(test_data)
  model.summary()

  # graph.plot_mse_per_time_step(model, test_data, args.pred_len, args.supplement, f'graph/{args.supplement}/{args.data}_{args.pred_len}/mse_per_time_step.png')


  if args.delete_checkpoint:
    for f in glob.glob(checkpoint_path + '*'):
      os.remove(f)

  # save result to csv
  data = {
      'data': [args.data],
      'supplement': [args.supplement],
      'pred_len': [args.pred_len],
      'mae': [test_result[0]],
      # 'val_mse': [history.history['val_loss'][best_epoch]],
      # 'train_mse': [history.history['loss'][best_epoch]],
      # 'training_time': elasped_training_time,
  }
  if 'TSMixer' in args.model:
    data['ff_dim'] = args.ff_dim

  df = pd.DataFrame(data)
  if os.path.exists(args.result_path):
    df.to_csv(args.result_path, mode='a', index=False, header=False)
  else:
    df.to_csv(args.result_path, mode='w', index=False, header=True)


if __name__ == '__main__':
  main()
