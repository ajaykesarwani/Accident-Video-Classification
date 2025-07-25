# Acident Video Classification

This repository contains two Jupyter notebooks demonstrating a convolutional neural network (CNN) for accident video classification, with and without a pretrained model approach. The model is designed to classify images into multiple classes using Keras and TensorFlow.

## 📁 Files

- `Pretrain Model.ipynb`: Defines and trains a CNN model and saves the weights for future use.
- `Final_CNN.ipynb`: Loads the pretrained model weights, fine-tunes the model, and evaluates performance on a test dataset.

---

## 🧠 Project Overview

The goal of this project is to build and improve an accident video classification model using a convolutional neural network. The workflow is divided into two stages:

1. **Pretraining Phase** (`Pretrain Model.ipynb`):
   - Define a CNN model architecture.
   - Train the model on a dataset.
   - Save the model weights to disk.

2. **Final Training & Evaluation** (`Final_CNN.ipynb`):
   - Load the saved pretrained weights.
   - Continue training or fine-tune on the same or new dataset.
   - Evaluate the model’s performance.

---
