from sklearn.cluster import KMeans
import numpy as np
import pdb

class FourierFeatureExtractor:
    def __init__(self, input_data, max_frequency_index=512):
        self.input_data = input_data
        self.max_frequency_index = max_frequency_index

    def initialize_fourier_parameters(self):
        fourier_features = {}
        for i in range(self.input_data.shape[1]):
            single_feature_data = self.input_data.iloc[:, i].values
            series_length = len(single_feature_data)
            fft_values = np.fft.rfft(single_feature_data)

            all_frequencies = np.arange(len(fft_values))
            valid_indices = all_frequencies[(series_length / all_frequencies) <= self.max_frequency_index]

            valid_power_spectrum = np.abs(fft_values[valid_indices]) ** 2
            top_indices = np.argsort(valid_power_spectrum)[-3:]

            top_components = []
            for index in top_indices:
                a = fft_values[index].real / series_length
                b = fft_values[index].imag / series_length
                k = valid_indices[index]

                top_components.append({'k': k, 'a': a, 'b': b})

            fourier_features[self.input_data.columns[i]] = top_components
        return fourier_features

def prepare_features(data_loader):
    # Fourier特徴抽出
    feature_extractor = FourierFeatureExtractor(data_loader.train_df)
    fourier = feature_extractor.initialize_fourier_parameters()

    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=5, random_state=0)
    kmeans.fit(data_loader.train_df.values)
    centers = kmeans.cluster_centers_ # n_clusters個のクラスタの座標

    return fourier, centers
