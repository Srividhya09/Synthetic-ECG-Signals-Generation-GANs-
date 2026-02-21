# ECG-GAN: Synthetic ECG Signal Generation using Deep Convolutional Generative Adversarial Networks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📋 Overview

ECG-GAN is a deep learning project that uses Deep Convolutional Generative Adversarial Networks (DCGAN) to generate synthetic Electrocardiogram (ECG) signals, specifically focused on Atrial Fibrillation (AF) heartbeats. This project is particularly useful for medical data augmentation, addressing the challenge of limited labeled ECG data in healthcare settings.

**Dataset Source:** [PhysioNet/CinC 2017 Challenge](https://www.kaggle.com/datasets/awsaf49/physionet-cinc-2017-dataset) - AF Classification from a Short Single Lead ECG Recording

## 🏗️ Project Architecture

```
ECG-GAN/
├── GAN_model.py                   # Main DCGAN model implementation
├── train.py                       # Training script
├── process_ecg.py                 # ECG data preprocessing
├── generate_synthetic_ecg.py      # Generate synthetic ECG signals
├── Minibatchdiscrimination.py     # Minibatch discrimination layer
├── requirement.txt                # Python dependencies
├── module/
│   ├── generator.py               # Generator architectures (G_v1 - G_v5)
│   └── discriminator.py           # Discriminator architectures (D_v1 - D_v3)
├── utils/
│   └── data_util.py               # Data loading and preprocessing utilities
├── robustness_analysis/
│   ├── robustness_analysis.py     # Robustness evaluation using DTW
│   └── analysis/
│       └── noise_utils.py         # Noise injection utilities
├── save_model/                    # Trained generator models
├── train_outputs_from_GAN_Model/  # Training progress and sample outputs
└── utils/                         # Utility functions
```

## 🚀 Features

- **Multiple Generator Architectures**: Support for 5 different generator versions (G_v1 - G_v5)
  - G_v1: BiLSTM + Conv1D (Proposed method)
  - G_v2: Dense neural network
  - G_v3: LSTM-based
  - G_v4: BiLSTM-based
  - G_v5: ecgGAN-inspired architecture

- **Multiple Discriminator Architectures**: Three discriminator variants (D_v1 - D_v3)

- **Minibatch Discrimination**: Helps prevent mode collapse during training

- **Flexible Noise Input**: Supports both random sine waves and standard normal distribution

- **Robustness Analysis**: Evaluate synthetic ECG quality using Dynamic Time Warping (DTW)

- **Model Checkpointing**: Save models at various training intervals

## 📊 Model Architecture Details

### Generator (Default - G_v1)
```
Input: Latent vector (100,)
  ↓
Reshape → BiLSTM(16, return_sequences=True)
  ↓
Conv1D(32, kernel_size=8) + LeakyReLU
  ↓
UpSampling1D → Conv1D(16, kernel_size=8) + LeakyReLU
  ↓
UpSampling1D → Conv1D(8, kernel_size=8) + LeakyReLU
  ↓
Conv1D(1, kernel_size=8) + Flatten
  ↓
Dense(180) + Activation(tanh)
  ↓
Output: ECG Signal (180, 1)
```

### Discriminator
```
Input: ECG Signal (180, 1)
  ↓
Conv1D(8) + LeakyReLU + MaxPooling1D(3)
  ↓
Conv1D(16) + MaxPooling1D(3)
  ↓
Conv1D(32) + MaxPooling1D(3)
  ↓
Conv1D(64) + MaxPooling1D(3)
  ↓
Flatten + Dense(1) + Sigmoid
  ↓
Output: Validity (0-1)
```

## 🛠️ Installation

1. **Clone the repository:**
```
bash
git clone https://github.com/yourusername/ECG-GAN.git
cd ECG-GAN
```

2. **Create a virtual environment (optional but recommended):**
```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```
bash
pip install -r requirement.txt
```

**Note:** Some dependencies may require specific versions. The project uses:
- TensorFlow 2.x
- NumPy, SciPy
- Matplotlib
- biosppy (for ECG signal processing)
- WFDB (for PhysioNet data)
- h5py
- tqdm
- fastdtw (for robustness analysis)

## 📁 Data Preparation

### Option 1: Download PhysioNet 2017 Dataset

1. Download the dataset from Kaggle:
   - [PhysioNet/CinC 2017 Dataset](https://www.kaggle.com/datasets/awsaf49/physionet-cinc-2017-dataset)

2. Extract the dataset to a folder named `training2017`

3. Update the paths in `process_ecg.py` if needed:
```python
AF_DATASET_DIR = 'path/to/training2017'
LABEL_PATH = 'path/to/training2017/REFERENCE-original.csv'
```

### Option 2: Use Pre-processed Data

The project includes pre-processed data files:
- `X_train_af.pkl` - Processed AF ECG signals
- `y_af.pkl` - Corresponding labels
- `AF_heartbeat.pkl` - Extracted heartbeats
- `ECG_data.pkl` - Full ECG dataset

## 🎯 Training the Model

### Run Training:
```
bash
python train.py
```

### Training Parameters (Default):
```
python
EPOCHS = 10000
LATENT_SIZE = 100
BATCH_SIZE = 128
INPUT_SHAPE = (180, 1)
SAVE_INTERVAL = 100
SAVE_MODEL_INTERVAL = 1000
RANDOM_SINE = False
SCALE = 2
MINIBATCH = True
GEN_VERSION = 0  # 0 = Default; 1-5 = Alternate versions
```

### Training Output:
- Model checkpoints saved to `save_model/`
- Sample ECG plots saved to `image/`
- Training progress saved to `output/progress_report.json`

## 🔬 Generating Synthetic ECG Signals

### Generate New Signals:
```
bash
python generate_synthetic_ecg.py
```

### Parameters:
```
python
num_signals = 100      # Number of signals to generate
latent_dim = 100       # Latent space dimensionality
generator_model_path = 'save_model/final_generator.h5'
```

## 📈 Robustness Analysis

Run robustness analysis to evaluate synthetic ECG quality:

```
bash
cd robustness_analysis
python robustness_analysis.py
```

This analysis:
- Loads real and synthetic ECG signals
- Injects Gaussian noise at various levels (0.0, 0.01, 0.05, 0.1)
- Computes DTW (Dynamic Time Warping) distances
- Visualizes robustness to noise

## 📊 Results

The trained models are saved in `save_model/`:
- `gen_1000.h5` - After 1000 epochs
- `gen_2000.h5` - After 2000 epochs
- ... (saved every 1000 epochs)
- `final_generator.h5` - Final trained model

Sample outputs are in `train_outputs_from_GAN_Model/`:
- `sample_*.csv` - Generated ECG samples
- `prob_*.csv` - Discriminator probabilities

## 🔧 Customization

### Use Different Generator Version:
```
python
dcgan = DCGAN(
    INPUT_SHAPE, LATENT_SIZE, 
    gen_version=1  # or 2, 3, 4, 5
)
```

### Use Minibatch Discrimination:
```
python
dcgan = DCGAN(
    INPUT_SHAPE, LATENT_SIZE, 
    minibatch=True
)
```

### Custom Training Parameters:
```
python
dcgan.train(
    epochs=5000,
    X_train=X_train,
    batch_size=64,
    save_interval=50,
    save_model_interval=500
)
```

## 📝 File Descriptions

| File | Description |
|------|-------------|
| `GAN_model.py` | Main DCGAN implementation with Generator and Discriminator |
| `train.py` | Training script with data loading and preprocessing |
| `process_ecg.py` | ECG signal processing and heartbeat extraction |
| `generate_synthetic_ecg.py` | Generate synthetic ECG using trained model |
| `Minibatchdiscrimination.py` | Minibatch discrimination layer (prevents mode collapse) |
| `module/generator.py` | Multiple generator architecture implementations |
| `module/discriminator.py` | Multiple discriminator architecture implementations |
| `utils/data_util.py` | Data loading and preprocessing utilities |
| `robustness_analysis/robustness_analysis.py` | DTW-based robustness analysis |

## 🏥 Medical Applications

This synthetic ECG generation system can be used for:

1. **Data Augmentation**: Generate more training samples for ECG classification models
2. **Privacy-Preserving Data Sharing**: Share synthetic data without exposing patient information
3. **Imbalanced Dataset Handling**: Generate more samples for underrepresented classes
4. **Model Testing**: Test ECG analysis algorithms with diverse synthetic data

## ⚠️ Limitations & Future Work

- Currently focused on Atrial Fibrillation (AF) detection
- Can be extended to other cardiac conditions
- Quality of generated signals depends on training data quality
- Further validation with clinical experts needed for medical applications

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- PhysioNet/CinC 2017 Challenge for the ECG dataset
- Reference paper: [Synthesis of Realistic ECG using Generative Adversarial Networks](https://arxiv.org/abs/1909.09150)

<p align="center">Made with ❤️ for Medical AI Research</p>
