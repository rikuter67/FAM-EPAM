from sklearn.cluster import KMeans
import numpy as np

def initialize_fourier_parameters(input_data, max_frequency_index=512):
    fourier_features = {}
    for i in range(input_data.shape[1]):
        single_feature_data = input_data.iloc[:, i].values
        series_length = len(single_feature_data)
        fft_values = np.fft.rfft(single_feature_data)

        all_frequencies = np.arange(len(fft_values))
        valid_indices = all_frequencies[(series_length / all_frequencies) <= max_frequency_index]
        valid_power_spectrum = np.abs(fft_values[valid_indices]) ** 2
        
        if input_data.shape[1] > 50:
            top_indices = np.argsort(valid_power_spectrum)[-1:]
        else:
            top_indices = np.argsort(valid_power_spectrum)[-3:]

        top_components = []
        for index in top_indices:
            a = fft_values[index].real / series_length
            b = fft_values[index].imag / series_length
            k = valid_indices[index]

            top_components.append({'k': k, 'a': a, 'b': b})

        fourier_features[input_data.columns[i]] = top_components
    return fourier_features

def prepare_features(data_loader, supplement=None):
    fourier, centers = None, None

    if supplement == 'FAM':
        fourier = initialize_fourier_parameters(data_loader.train_df)

    elif supplement == 'EPAM':
        kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
        kmeans.fit(data_loader.train_df.values)
        centers = kmeans.cluster_centers_

    return fourier, centers
