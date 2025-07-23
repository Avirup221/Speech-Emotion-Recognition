Speech Emotion Recognition
This repository contains a Jupyter Notebook (designed for Google Colab) that implements a Speech Emotion Recognition (SER) system. The project focuses on processing audio data, extracting relevant features, training a deep learning model using PyTorch, and evaluating its performance in classifying emotions from speech.

Table of Contents
Project Overview

Features

Datasets

Prerequisites

Setup and Usage

Results

Future Work

License

Contact

Project Overview
Speech Emotion Recognition is a challenging task that involves identifying the emotional state of a speaker from their voice. This project tackles SER by:

Data Collection & Preprocessing: Handling raw audio files from multiple datasets.

Feature Engineering: Extracting meaningful features (like MFCCs) that represent the emotional content of speech.

Deep Learning Model: Building and training a neural network (likely CNN/RNN based, given the PyTorch imports) to classify emotions.

Evaluation: Assessing the model's accuracy, precision, recall, and F1-score.

Features
Google Colab Integration: Designed for easy execution within Google Colab, including seamless Google Drive mounting for dataset access.

Multi-Dataset Support: Capable of handling and combining data from various speech emotion datasets. The notebook explicitly mentions:

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

SAVEE (Surrey Audio-Visual Expressed Emotion)

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

Audio Preprocessing: Utilizes librosa for loading and manipulating audio signals.

Feature Extraction: Extracts Mel-frequency Cepstral Coefficients (MFCCs), a common feature set for audio analysis.

Data Augmentation: Includes steps for augmenting the dataset to improve model robustness (implied by augmented_data.csv).

Data Scaling & Encoding: Standardizes features and one-hot encodes emotion labels for model training.

PyTorch Framework: Implements the deep learning model using PyTorch, leveraging nn.Module, optim, DataLoader, and TensorDataset.

Comprehensive Evaluation: Provides detailed classification metrics, including accuracy, precision, recall, and F1-score, along with confusion matrices.

Datasets
The project uses the following speech emotion datasets:

RAVDESS: A professional audio-visual dataset of emotional speech and song.

SAVEE: An audio-visual emotional speech database.

CREMA-D: A crowd-sourced emotional multimodal actors dataset.

Note: The notebook expects these datasets to be available as .zip files within a Datasets folder in your Google Drive (e.g., /content/drive/My Drive/Datasets/ravdess.zip).

Prerequisites
To run this notebook, you will need:

Python: Version 3.x (typically pre-installed in Google Colab).

Google Colab Account: For easy execution and access to GPU resources.

Required Python Libraries:

os

sys

numpy

pandas

librosa

librosa.display

seaborn

matplotlib.pyplot

zipfile

google.colab

IPython.display

re

sklearn (StandardScaler, OneHotEncoder, LabelEncoder, confusion_matrix, classification_report, train_test_split)

soundfile

torch

torch.nn

torch.optim

torch.nn.functional

torch.utils.data (Dataset, DataLoader, TensorDataset)

These libraries are typically pre-installed in Google Colab environments, or can be installed using pip if necessary.

Setup and Usage
Open in Google Colab: Upload the Speech_Emotion_Recognition_now.ipynb file to your Google Drive and open it with Google Colab.

Mount Google Drive: Run the cell that mounts your Google Drive to access the datasets. You will be prompted to authorize Google Drive access.

from google.colab import drive
drive.mount('/content/drive')

Place Datasets: Ensure your zipped datasets (e.g., ravdess.zip, savee.zip, CREMA-D.zip) are placed in a folder named Datasets within your Google Drive's "My Drive" (i.e., /content/drive/My Drive/Datasets/).

Extract Datasets: Run the cells that extract the datasets to the Colab environment. The notebook specifically extracts ravdess.zip to /content/ravdess. You might need to adapt this for other datasets.

zip_path = "/content/drive/My Drive/Datasets/ravdess.zip"
extract_path = "/content/ravdess"
os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("ravdess dataset is extracted successfully!")

Run All Cells: Execute all cells in the notebook sequentially. This will perform data loading, preprocessing, feature extraction, model training, and evaluation.

Results
The notebook will output various results, including:

Dataset statistics (e.g., Ravdess_df.shape, Ravdess_df.head()).

Sample audio playback.

Visualizations of audio waveforms and spectrograms (MFCCs).

Detailed classification metrics (Accuracy, Precision, Recall, F1 Score).

A comprehensive classification report per class.

Confusion matrices for model performance visualization.

Future Work
Explore More Features: Experiment with other audio features like Chroma, Tonal Centroid (Centroid), Zero-Crossing Rate (ZCR), and Root Mean Square Energy (RMSE).

Advanced Models: Implement more complex deep learning architectures (e.g., LSTMs, GRUs, or transformer-based models) for potentially higher accuracy.

Hyperparameter Tuning: Optimize model hyperparameters using techniques like Grid Search or Random Search.

Real-time Inference: Develop a real-time emotion recognition system.

Larger Datasets: Integrate and test with larger and more diverse datasets for improved generalization.

Cross-Corpus Evaluation: Evaluate model performance across different datasets to assess generalization capabilities.

License
This project is open-source and available under the MIT License.
