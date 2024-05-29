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

## Getting Started

We separate our codes for supervised learning and self-supervised learning into two folders: `supervised_learning` and `self_supervised_learning`. Please choose the one that you want to work with.

### Supervised Learning

1. Install requirements.
    ```bash
    pip install -r requirements.txt
    ```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a separate folder `./dataset` and put all the CSV files in the directory.

3. Training. All the scripts are in the directory `./scripts/PatchTST`. The default model is TSMixer with FAM and EPAM. For example, if you want to get the multivariate forecasting results for the weather dataset, just run the following command, and you can open `./result.txt` to see the results once the training is done:
    ```bash
    sh ./scripts/PatchTST/weather.sh
    ```

You can adjust the hyperparameters based on your needs (e.g., different patch lengths, different look-back windows, and prediction lengths). We also provide codes for the baseline models.

### Self-supervised Learning

1. Follow the first 2 steps above.

2. Pre-training: The script `patchtst_pretrain.py` is to train the TSMixer with FAM and EPAM. To run the code with a single GPU on ettm1, just run the following command:
    ```bash
    python patchtst_pretrain.py --dset ettm1 --mask_ratio 0.4
    ```
    The model will be saved to the `saved_model` folder for the downstream tasks. There are several other parameters that can be set in the `patchtst_pretrain.py` script.

3. Fine-tuning: The script `patchtst_finetune.py` is for the fine-tuning step. Either linear probing or fine-tuning the entire network can be applied.
    ```bash
    python patchtst_finetune.py --dset ettm1 --pretrained_model <model_name>
    ```

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