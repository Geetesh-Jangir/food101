
# 🍔 Food-101 Classification using Transfer Learning (TensorFlow)

This repository contains a single Jupyter notebook: **`food-101.ipynb`**, which fine-tunes a pre-trained deep learning model using **Transfer Learning** on the [Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

The model is trained using **TensorFlow**, **Keras**, and leverages best practices like data prefetching, learning rate scheduling, checkpointing, and early stopping.

---

## 📌 Notebook Summary

- **Notebook**: `food-101.ipynb`
- **Dataset**: Food-101 (manually downloaded and extracted)
- **Model**: Pre-trained (e.g., EfficientNetB0 or similar) with fine-tuning
- **Accuracy Achieved**: **83.3%** on validation set
- **Environment**: Google Colab with GPU support

---

## 🧠 Features in the Notebook

- Uses `image_dataset_from_directory` with validation split
- Preprocessing with `cache`, `shuffle`, `batch`, and `prefetch`
- Implements:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint` (saving `.h5` weights)
- Fine-tunes a pre-trained model on Food-101 categories
- Saves and loads best model weights for inference
- Includes evaluation and model summary

---

## 🗃 Dataset Used

- **Food-101**
- Source: [ETH Zurich](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- Used local path to manually extracted images
- Format: `images/class_name/image.jpg`

---

## 📦 Requirements

```bash
tensorflow >= 2.12
numpy
matplotlib
