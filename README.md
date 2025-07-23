## Speech Emotion Recognition (SER)
This repository contains a Jupyter Notebook (designed for Google Colab) that implements a Speech Emotion Recognition (SER) system. The project focuses on processing raw audio data, extracting relevant features, training a deep learning model using PyTorch, and evaluating its performance in classifying emotions from speech.

## 📌 Table of Contents
Project Overview

Motivation

Features

Datasets

Prerequisites

Installation

How to Run

Model Architecture (Implied)

Evaluation Metrics

Results

Future Work

License

Contact

## 🧠 Project Overview
Speech Emotion Recognition (SER) is a challenging task that involves identifying the emotional state of a speaker from their voice. This project aims to develop a robust SER system by leveraging deep learning techniques on various emotional speech datasets. The system processes audio, extracts meaningful features, and then classifies the emotional content.

## 💡 Motivation
Emotion recognition from speech has numerous applications, including:

Human-Computer Interaction: Creating more natural and empathetic AI assistants.

Mental Health Monitoring: Detecting early signs of emotional distress or depression.

Call Centers: Analyzing customer sentiment for improved service.

Security: Identifying emotional states in critical situations.

This project serves as a foundational step towards building more sophisticated and real-world SER applications.

## ✨ Features
Google Colab Integration: Designed for easy execution within Google Colab, leveraging its free GPU resources and seamless Google Drive mounting for dataset access.

Multi-Dataset Support: Capable of handling and combining data from multiple popular speech emotion datasets.

Audio Preprocessing: Utilizes librosa for efficient loading, resampling, and manipulation of audio signals.

Feature Extraction: Extracts Mel-frequency Cepstral Coefficients (MFCCs), a widely used and effective feature set for speech analysis, capturing the timbre and spectral characteristics of the voice.

Data Augmentation: Includes techniques to augment the dataset, increasing its diversity and helping to prevent overfitting, thereby improving model generalization.

Data Scaling & Encoding: Standardizes extracted features using StandardScaler and one-hot encodes emotion labels using OneHotEncoder for compatibility with neural networks.

Deep Learning with PyTorch: Implements a neural network model using the PyTorch framework, leveraging nn.Module, optim, DataLoader, and TensorDataset for efficient training and inference

Comprehensive Evaluation: Provides detailed classification metrics, including accuracy, precision, recall, F1-score, and confusion matrices, to thoroughly assess model performance.

Audio Playback: Includes functionality to play sample audio files directly within the notebook for quick verification of data.

## 📂 Datasets
The project is designed to work with (and combine) the following publicly available speech emotion datasets:

RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song): Contains emotional speech and song from 24 professional actors (12 male, 12 female), expressing 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprise) in two emotional intensities.

SAVEE (Surrey Audio-Visual Expressed Emotion): Features recordings from 4 male actors, expressing 7 emotions (anger, disgust, fear, happiness, sadness, surprise, and neutral).

CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset): Includes 7,442 original clips from 91 actors (48 male, 43 female) expressing 6 emotions (anger, disgust, fear, happiness, neutral, and sadness) at various intensity levels.

## Note: The notebook is configured to access these datasets as zipped files within a Datasets folder in your Google Drive (e.g., /content/drive/My Drive/Datasets/ravdess.zip).

🛠 Prerequisites
To run this notebook, you will need:

Python: Version 3.x (typically pre-installed in Google Colab).

Google Colab Account: For easy execution and access to GPU resources.

## Required Python Libraries:

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

warnings

## 🧰 Installation
First, clone the repository:

## git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition

The required Python libraries are typically pre-installed in Google Colab environments. If you are running this locally, you can install them via pip:

pip install -r requirements.txt

requirements.txt content:

numpy
pandas
librosa
seaborn
matplotlib
scikit-learn
torch
soundfile

## 🚀 How to Run
Open in Google Colab: Upload the Speech_Emotion_Recognition_now.ipynb file to your Google Drive and open it with Google Colab.

Mount Google Drive: Run the cell that mounts your Google Drive to access the datasets. You will be prompted to authorize Google Drive access.

from google.colab import drive
drive.mount('/content/drive')

Place Datasets: Ensure your zipped datasets (e.g., ravdess.zip, savee.zip, CREMA-D.zip) are placed in a folder named Datasets within your Google Drive's "My Drive" (i.e., /content/drive/My Drive/Datasets/).

Extract Datasets: Run the cells that extract the datasets to the Colab environment. The notebook specifically extracts ravdess.zip to /content/ravdess. You might need to adapt this for other datasets by duplicating and modifying the extraction cells.

# Example for RAVDESS:
import os
import zipfile

zip_path = "/content/drive/My Drive/Datasets/ravdess.zip"
extract_path = "/content/ravdess" # Or a suitable path for other datasets

os.makedirs(extract_path, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(f"{os.path.basename(zip_path)} dataset is extracted successfully!")

Run All Cells: Execute all cells in the notebook sequentially. This will perform data loading, preprocessing, feature extraction, model training, and evaluation.


## 📊 Evaluation Metrics
The notebook evaluates the model's performance using standard classification metrics:

Accuracy: The proportion of correctly classified instances.

Precision: The ratio of true positives to the sum of true positives and false positives. Useful when the cost of false positives is high.

Recall (Sensitivity): The ratio of true positives to the sum of true positives and false negatives. Useful when the cost of false negatives is high.

F1-Score: The harmonic mean of precision and recall, providing a balanced measure.

Confusion Matrix: A table summarizing the performance of a classification algorithm, showing true positives, true negatives, false positives, and false negatives for each class.

Classification Report: A detailed report showing precision, recall, and F1-score for each class, along with support (number of occurrences of each class).

## 🏆 Results
The notebook will output various results, including:

Dataset statistics (e.g., DataFrame shapes, head of dataframes).

Sample audio playback for verification.

Visualizations of audio waveforms and spectrograms (MFCCs).

Detailed classification metrics (Accuracy, Precision, Recall, F1 Score).

A comprehensive classification report per class.

Confusion matrices for model performance visualization.

(Note: Specific numerical results are not included here as they depend on the execution of the notebook and the training process.)

## 🔮 Future Work
Explore More Features: Experiment with other audio features like Chroma, Tonal Centroid (Centroid), Zero-Crossing Rate (ZCR), and Root Mean Square Energy (RMSE) to see their impact on model performance.

Advanced Deep Learning Models: Implement more complex deep learning architectures (e.g., transformer-based models, attention mechanisms) for potentially higher accuracy and better capture of long-range dependencies.

Hyperparameter Tuning: Systematically optimize model hyperparameters (learning rate, batch size, number of layers, neurons, etc.) using techniques like Grid Search, Random Search, or Bayesian Optimization.

Real-time Inference: Develop a lightweight version of the model suitable for real-time emotion recognition from live audio streams.

Larger and More Diverse Datasets: Integrate and test with larger and more diverse datasets to improve model generalization and robustness across different speakers, languages, and recording conditions.

Cross-Corpus Evaluation: Evaluate model performance across different datasets (training on one, testing on another) to assess generalization capabilities and identify domain adaptation challenges.

Multimodal Fusion: Incorporate other modalities like facial expressions or text transcripts for multimodal emotion recognition, potentially leading to more accurate and robust systems.

Uncertainty Quantification: Implement methods to quantify the model's uncertainty in its predictions, providing more reliable diagnostic information.

