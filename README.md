# EPAM_and_FAM_FeatureMixing_TSMixer

### This is an official implementation of the paper "Permutation Dependent Feature Mixing for Multivariate Time Series Forecasting" presented at ECML 2024.

## Key Designs

:star2: **Frequency-Aware Mixer (FAM)**: Enhances feature mixing by incorporating frequency information, improving the model's ability to capture periodic patterns in the data.

:star2: **Event Proximity-Aware Mixer (EPAM)**: Focuses on the temporal proximity of events, enabling the model to better understand and predict time-dependent relationships.

## Results

### Performance Comparison

The enhanced TSMixer model with FAM and EPAM demonstrates significant improvements in forecasting accuracy compared to the baseline TSMixer and other state-of-the-art models.

![Performance Comparison](https://github.com/rikuter67/EPAM_and_FAM_FeatureMixing_TSMixer/main/pic/performance_comparison.png)

### Visualizations

The following visualizations illustrate the effectiveness of our enhanced feature mixing approach.

![Visualization 1](https://github.com/rikuter67/EPAM_and_FAM_FeatureMixing_TSMixer/main/pic/visualization1.png)
![Visualization 2](https://github.com/rikuter67/EPAM_and_FAM_FeatureMixing_TSMixer/main/pic/visualization2.png)

## Running the Experiments

To run the experiments with tuned hyperparameters, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/rikuter67/EPAM_and_FAM_FeatureMixing_TSMixer.git
    cd EPAM_and_FAM_FeatureMixing_TSMixer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the datasets from the [Autoformer GitHub repository](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) and place them in the `./dataset` directory.

4. Run the experiments using the provided script:
    ```bash
    ./run_tuned_hparam.sh
    ```

This will execute the experiments with the specified datasets and prediction lengths, using the tuned hyperparameters defined in the script.



### File Structure

The repository is organized as follows:

```plaintext
.
├── README.md
├── __pycache__
│   ├── data_loader.cpython-311.pyc
│   ├── data_loader.cpython-38.pyc
│   ├── data_loader.cpython-39.pyc
│   ├── graph.cpython-39.pyc
│   └── utils.cpython-39.pyc
├── checkpoints
│   ├── <model_checkpoints>  # Model checkpoints for different datasets and prediction lengths
├── data_loader.py  # Script to load and preprocess the data
├── dataset
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── all_six_datasets.zip
│   ├── electricity.csv
│   ├── exchange_rate.csv
│   ├── national_illness.csv
│   ├── traffic.csv
│   └── weather.csv
├── graph.py  # Script for plotting and visualization
├── models
│   ├── __init__.py
│   ├── cnn.py
│   ├── fmix_only.py
│   ├── full_linear.py
│   ├── linear.py
│   ├── rev_in.py
│   ├── tmix_only.py
│   ├── tsmixer.py
│   └── tsmixer_rev_in.py
├── requirements.txt  # Required dependencies
├── result.csv  # File to store the results
├── run.py  # Main script to run the model
├── run_tuned_hparam.sh  # Script to run experiments with tuned hyperparameters
├── run_tuned_hparam_1.sh  # Additional script for running experiments
└── utils.py  # Utility functions

## Explanation of Key Files and Directories
- README.md: This file, providing an overview of the project and instructions for setup and usage.
- __pycache__: Directory containing Python bytecode files for performance optimization.
- checkpoints: Directory where trained model checkpoints are saved. Each checkpoint corresponds to a different dataset and prediction length.
- data_loader.py: Script responsible for loading and preprocessing the dataset.
- dataset: Directory where datasets should be placed. Contains CSV files for various datasets used in the experiments.
- graph.py: Script for generating plots and visualizations of the results.
- models: Directory containing the implementation of different model variants, including TSMixer and its enhanced versions with FAM and EPAM.
- requirements.txt: File listing the Python dependencies required for running the code.
- result.csv: File where detailed results of each run are appended.
- run.py: Main script to execute the model training and evaluation.
- run_tuned_hparam.sh: Script to run experiments with predefined hyperparameters.
- utils.py: Utility functions used throughout the project.

## Results
- The results of the experiments, including the trained model checkpoints, will be stored in the checkpoints directory.
- Detailed results for each run will be appended to the result.csv file.

## Acknowledgement

We appreciate the following GitHub repos very much for the valuable code base and datasets:

- [TSMixer](https://github.com/ts-kim/TSMixer)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
- [Informer2020](https://github.com/zhouhaoyi/Informer2020)
- [Autoformer](https://github.com/thuml/Autoformer)
- [RevIN](https://github.com/ts-kim/RevIN)

## Contact

If you have any questions or concerns, please contact us: s256279@wakayama-u.ac.jp or 先生のメールアドレス or submit an issue.

## Citation

If you find this repo useful in your research, please consider citing our paper as follows:

@inproceedings{yamazono2024tsmixer,
title={Permutation Dependent Feature Mixing for Multivariate Time Series Forecasting},
author={R. Yamazono and H. Hachiya},
booktitle={ECML 2024},
year={2024}
}