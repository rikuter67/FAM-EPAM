import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pdb
import numpy as np
import os
from models.tsmixer import ResBlock
from models.tmix_only import ResBlockT
from models.fmix_only import ResBlockF


def predictions_with_history(test_data, data, model, seq_len, pred_len, save_path):
    if 'ETT' in data:
        all_feature_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        selected_feature_names = ['OT']
    elif data == 'weather':
        all_feature_names = ['p (mbar)','T (degC)','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)','VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)','wv (m/s)','max. wv (m/s)','wd (deg)','rain (mm)','raining (s)','SWDR (W/m^2)','PAR (µmol/m^2/s)','max. PAR (µmol/m^2/s)','Tlog (degC)','CO2']
        selected_feature_names = ['T (degC)', 'wv (m/s)', 'rain (mm)', 'SWDR (W/m^2)', 'CO2']
    else:
        all_feature_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    # 選択された特徴量のインデックスを取得
    selected_feature_indices = [all_feature_names.index(name) for name in selected_feature_names]

    batch_index = 0

    # テストデータの各バッチに対してループ
    for x, y in test_data:
        test_x, test_y = x, y

        # モデルによる予測
        pred_y = model.predict(test_x)

        plt.figure(figsize=(10, 8))

        for i, feature_index in enumerate(selected_feature_indices):
            plt.subplot(len(selected_feature_names), 1, i + 1)
            # 履歴と真値を連続してプロット
            full_history = tf.concat([test_x[0, :, feature_index], test_y[0, :, feature_index]], axis=0)
            plt.plot(full_history, label=f'{selected_feature_names[i]} - History & True', linewidth=3)
            # 予測値をプロット
            plt.plot(range(seq_len, seq_len + pred_len), pred_y[0, :, feature_index], label=f'{selected_feature_names[i]} - Predicted', linewidth=3)
            plt.axvline(x=seq_len, color='gray', linewidth=1.0)
            # plt.title(selected_feature_names[i], fontsize=28)
            # plt.legend(loc='upper left')

        plt.tight_layout()
        new_save_path = f'{save_path}_{batch_index}.pdf'
        plt.savefig(new_save_path)
        plt.close()

        batch_index += 1


def history(history, save_path='graph/train_valid.png'):
  # 訓練履歴を表示
  plt.figure(figsize=(12, 6))
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title("Training History")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(save_path)  # 画像として保存
  plt.close()


class PlotLayerOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, layer_names, test_data):
        super(PlotLayerOutputCallback, self).__init__()
        self.layer_names = layer_names  # 層名のリスト
        self.test_data = test_data

    def on_test_end(self, logs=None):
        for layer_name in self.layer_names:
            layer_output_model = tf.keras.Model(inputs=self.model.input,
                                                outputs=self.model.get_layer(layer_name).output)
            layer_output = layer_output_model.predict(self.test_data)
            self.plot(layer_output, layer_name)

    def plot(self, data, name): # arg.data, arg.supplement?
        labels = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']  # 凡例のラベル
        num_samples = data.shape[0]  # サンプルの数
        num_features = data.shape[2]  # 特徴の数（この場合は7）
        # 最初のバッチのデータを取得
        first_batch_data = data[0, :, :]  # 形状は (512, 7)

        plt.figure(figsize=(10, 4 * num_features))

        for feature_index in range(num_features):
            plt.subplot(num_features, 1, feature_index + 1)
            plt.plot(first_batch_data[:, feature_index])
            plt.title(f'{labels[feature_index]}')  # ラベルをタイトルに追加
            plt.ylim(-8, 8)
            plt.legend([labels[feature_index]], loc='upper right')  # ラベルを凡例に追加

        plt.tight_layout()
        plt.savefig(f'graph/inner_layer/{name}.png')
        plt.close()

def plot_res_block_weights(model, save_dir, data_name, pred_len, layer_name, time_phase):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ResBlockF):
            # ResBlock内の指定されたレイヤーの重みを取得
            target_layer = getattr(layer, layer_name, None)
            if target_layer is not None:
                weights, biases = target_layer.get_weights()
                plt.figure(figsize=(10, 8))
                sns.heatmap(weights, annot=False, cmap='RdBu_r')
                # ヒートマップの色のスケールを調節
                plt.clim(-0.5, 0.5)
                plt.title(f'{layer.name}_{layer_name}(data: {data_name}, Pred Len: {pred_len})')
                save_path = os.path.join(save_dir, f'{data_name}_predlen{pred_len}_{layer.name}_{layer_name}_{time_phase}.png')
                plt.savefig(save_path)
                plt.close()
                print(f'重みのヒートマップを{save_path}に保存しました')

def plot_last_dense_layer_weights(model, save_dir, data_name, pred_len):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights, annot=False, cmap='RdBu_r')
            plt.title(f'{layer.name}(data: {data_name}, Pred Len: {pred_len})')
            save_path = os.path.join(save_dir, f'{data_name}_predlen{pred_len}_{layer.name}_dense.png')
            plt.savefig(save_path)
            plt.close()
            print(f'重みのヒートマップを{save_path}に保存しました')

def plot_mse_per_time_step(model, test_data, pred_len, supplement, save_path):
    # モデルの予測を行う
    predictions = []
    actuals = []
    for x, y in test_data:
        y_pred = model.predict(x)
        predictions.append(y_pred)
        actuals.append(y)

    # numpy配列に変換
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # 時点ごとのMSEを計算
    mse_per_timepoint = []
    for i in range(pred_len):
        mse = np.mean((predictions[:, i, :] - actuals[:, i, :]) ** 2)
        mse_per_timepoint.append(mse)

    # 折れ線グラフのプロット
    plt.figure(figsize=(10, 6))

    if supplement == 'frequence':
        plt.plot(range(pred_len), mse_per_timepoint, marker='o', color='blue')
    elif supplement == 'similarity':
        plt.plot(range(pred_len), mse_per_timepoint, marker='o', color='orange')
    else:
        plt.plot(range(pred_len), mse_per_timepoint, marker='o', color='gray')
        
    plt.ylim(0, 0.7)
    plt.xlabel('Time Steps')
    plt.ylabel('MSE')
    plt.title('MSE at Each Prediction Step')
    plt.savefig(save_path)
    plt.close()

def plot_res_block_io(model, test_data, block_index, save_path):
    # 中間層の出力リストを作成（入力ノードを持つレイヤーのみ）
    layer_outputs = []
    for layer in model.layers[:block_index + 1]:
        if hasattr(layer, 'inbound_nodes') and layer.inbound_nodes:
            layer_outputs.append(layer.output)

    # 最初のレイヤーの入力を使用して新しい中間モデルを作成
    first_layer_input = model.layers[0].input
    
    activation_model = tf.keras.models.Model(inputs=first_layer_input, outputs=layer_outputs)

    # テストデータからサンプルを取得
    for x, y in test_data.take(1):
        # 中間層の出力を取得
        activations = activation_model.predict(x)

        # 対象レイヤーの入力と出力を取得
        layer_input = activations[block_index - 1] if block_index > 0 else x
        layer_output = activations[block_index]

        # プロット
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f'ResBlock {block_index} Input')
        plt.imshow(layer_input[0])  # 最初のサンプルの入力をプロット
        plt.subplot(1, 2, 2)
        plt.title(f'ResBlock {block_index} Output')
        plt.imshow(layer_output[0])  # 最初のサンプルの出力をプロット
        plt.savefig(save_path)
        plt.close()

class ResBlockIOCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, test_data, res_block_names, save_path):
        self.model = model
        self.test_data = test_data
        self.res_block_names = res_block_names
        self.save_path = save_path

    def on_test_begin(self, logs=None):
        # ResBlockの入力と出力を取得するためのモデルを作成
        self.io_models = []
        for name in self.res_block_names:
            res_block_layer = self.model.get_layer(name)
            io_model = tf.keras.Model(inputs=self.model.input, 
                                      outputs=[res_block_layer.input, res_block_layer.output])
            self.io_models.append(io_model)

    def on_test_end(self, logs=None):
        for i, (io_model, name) in enumerate(zip(self.io_models, self.res_block_names)):
            for x, y in self.test_data:
                inputs, outputs = io_model.predict(x)
                # ここで inputs と outputs をプロットまたは保存
                self.plot_io(inputs, outputs, i, name)

    def plot_io(self, inputs, outputs, index, layer_name):
        # 入力と出力をプロットする簡単な例
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(inputs[0])  # 最初のサンプルの入力をプロット
        plt.title(f'{layer_name} Input')
        plt.subplot(1, 2, 2)
        plt.plot(outputs[0])  # 最初のサンプルの出力をプロット
        plt.title(f'{layer_name} Output')
        plt.savefig(f'{self.save_path}_{layer_name}_{index}.png')
        plt.close() # 画像を保存して閉じる


