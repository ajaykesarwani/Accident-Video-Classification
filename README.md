# 🚗 Prediction of Traffic Accidents in DashCam Videos

This repository contains a deep learning project focused on detecting traffic accidents from dashcam video footage using convolutional neural networks (CNNs) and long short-term memory networks (LSTMs). The approach includes both training a model from scratch and leveraging pretrained models for improved accuracy.

---

## 🧠 Project Overview

The primary goal of this project is to **classify traffic accidents** in dashcam footage for applications in **autonomous driving safety systems**. The workflow is divided into two main phases:

1. **Pretraining Phase** (`Pretrain Model.ipynb`):
   - Defines a custom CNN architecture using TensorFlow and Keras.
   - Trains the model on a prepared dataset of images extracted from dashcam videos.
   - Saves model weights for later reuse.

2. **Fine-tuning & Evaluation Phase** (`Final_CNN.ipynb`):
   - Loads the pretrained weights.
   - Fine-tunes the model to improve generalization and accuracy.
   - Evaluates model performance on a test dataset.

Additionally, an LSTM-based model was explored to capture **temporal dependencies** across frames in videos. This model achieved the **highest classification accuracy of 82%**, demonstrating its effectiveness for real-time accident detection.

---

## 🏗️ Data Pipeline

- **Dashcam videos** were processed by extracting image frames using OpenCV.
- The image sequences were used to train both static (CNN-based) and temporal (LSTM-based) models.
- Data augmentation techniques were applied to improve robustness.

---

## 📁 Files

- `Pretrain Model.ipynb` – Trains and saves a CNN model from scratch.
- `Final_CNN.ipynb` – Loads pretrained weights, fine-tunes the model, and evaluates it.

---

## 📊 Results

- **CNN Model (with fine-tuning)**: Competitive baseline accuracy.
- **LSTM Model**: Best performance with **82% classification accuracy**, demonstrating the importance of modeling temporal relationships in video.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow** & **Keras**
- **OpenCV** (for video-to-frame conversion)
- **CNN** for image-based classification
- **LSTM** for sequence modeling
- **MobileNetV2** for transfer learning experiments

---

## 🚀 Future Work

- Improve real-time detection speed for deployment.
- Experiment with attention-based video models (e.g., Transformers).
- Deploy as an edge model for onboard vehicle systems.
