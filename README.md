# 🌿 Crop Disease Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-ResNet50-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)

**AI-powered plant disease detection system achieving up to 99%+ accuracy across 9 critical crops using custom CNNs and ResNet-50 transfer learning.**

</div>

---

## 📖 Overview

This project builds an end-to-end **plant disease classification system** that diagnoses diseases in 9 critical crops by analyzing leaf microscopy images. Each crop has its own dedicated Jupyter notebook, enabling independent training, evaluation, and model export.

The system covers both:
- **Custom CNN architectures** trained from scratch on crop-specific datasets
- **ResNet-50 transfer learning** with fine-tuned classification heads for higher accuracy

---

## 🎯 Supported Crops & Notebooks

| Crop | Notebook | Architecture | Notes |
|---|---|---|---|
| 🍎 Apple | `apple.ipynb` | Custom CNN | Train/val split internally |
| 🍎 Apple | `apple-transfer_learning.ipynb` | ResNet-50 | Transfer learning variant |
| 🍒 Cherry | `cherry.ipynb` | Custom CNN | Train/val split 70/30 |
| 🌽 Corn (Maize) | `corn.ipynb` | ResNet-50 | EarlyStopping callback |
| 🍇 Grape | `grape-transfer-learning.ipynb` | ResNet-50 | Manual 80/10/10 split |
| 🌶️ Pepper Bell | `pepper_bell.ipynb` | ResNet-50 | Data augmentation pipeline |
| 🥔 Potato | `potato.ipynb` | ResNet-50 | Full 80/10/10 split |
| 🍓 Strawberry | `strawberry.ipynb` | Custom CNN | 80/10/10 split |
| 🎋 Sugarcane | `sugarcane.ipynb` | ResNet-50 | Mixed architecture |
| 🍅 Tomato | `tomato.ipynb` | Custom CNN | Lightweight model |
| 🌱 Multi-Crop | `crop_diasease.ipynb` | Custom CNN | General crop baseline |

---

## 🧠 Problem Statement

> Manual identification of crop diseases by farmers is slow, inconsistent, and requires expert agronomists. This system provides an automated, image-based diagnostic tool that can identify disease from a single leaf photograph — enabling early intervention and reducing crop losses.

**Type:** Multi-class Image Classification  
**Domain:** Agricultural AI / Computer Vision  
**Difficulty:** Intermediate

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow 2.x, Keras |
| Transfer Learning | ResNet-50 (ImageNet pretrained) |
| Data Pipeline | `tf.keras.utils.image_dataset_from_directory` |
| Augmentation | `tf.keras.layers.RandomFlip`, `RandomRotation` |
| Callbacks | `EarlyStopping(restore_best_weights=True)` |
| Environment | Google Colab (GPU runtime) |
| Serialization | `.keras` model format |

---

## 🔧 Model Architectures

### Custom CNN (apple, cherry, strawberry, tomato, crop_diasease)

```
Input (128×128×3)
  → Conv2D(64, 3×3, relu) + MaxPool
  → Conv2D(128, 3×3, relu) + MaxPool
  → Conv2D(256, 3×3, relu) + MaxPool
  → Flatten → Dense(512, relu) → Dropout(0.2)
  → Dense(N_classes, softmax)
```

### ResNet-50 Transfer Learning (corn, grape, pepper_bell, potato, sugarcane, apple-TL)

```
Input (128×128×3)
  → ResNet-50 (ImageNet weights, frozen base)
  → GlobalAveragePooling2D → Flatten
  → Dense(512, relu) → Dropout(0.2)
  → Dense(256, relu) → Dropout(0.1)
  → Dense(N_classes, softmax)

Optimizer: Adam (lr=0.0001)
Loss: categorical_crossentropy
```

---

## 📊 Data Pipeline

Each notebook uses the modern TensorFlow data API:

```python
# Load with automatic train/val split
train_data, val_data = tf.keras.utils.image_dataset_from_directory(
    'crop_folder/',
    label_mode='categorical',
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset='both',
    seed=42,
)

# Optimised pipeline
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# In-model augmentation (training only)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
])
```

---

## 📁 Project Structure

```
plant-disease-prediction/
│
├── crop_diasease.ipynb              ← General multi-crop CNN
├── apple.ipynb                      ← Apple CNN
├── apple-transfer_learning.ipynb    ← Apple ResNet-50
├── cherry.ipynb                     ← Cherry CNN
├── corn.ipynb                       ← Corn ResNet-50
├── grape-transfer-learning.ipynb    ← Grape ResNet-50
├── pepper_bell.ipynb                ← Pepper Bell ResNet-50
├── potato.ipynb                     ← Potato ResNet-50
├── strawberry.ipynb                 ← Strawberry CNN
├── sugarcane.ipynb                  ← Sugarcane ResNet-50
├── tomato.ipynb                     ← Tomato CNN
└── README.md
```

Each notebook produces (after training):
- `<crop>_model.keras` — saved Keras model
- `<crop>_hist.json` — training history (loss/accuracy per epoch)

---

## ⚡ Running the Notebooks

### Option A — Google Colab (Recommended)

1. Upload the notebook and your dataset folder to Google Drive
2. Mount Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the dataset path in Cell 2 to your Drive location
4. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
5. Run all cells (`Runtime → Run all`)

### Option B — Local (GPU required for ResNet-50 notebooks)

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
jupyter notebook apple.ipynb
```

---

## 🗂️ Dataset Structure

Each notebook expects its dataset in a folder named after the crop:

```
apple/
├── apple_scab/
│   ├── image001.jpg
│   └── ...
├── apple_black_rot/
│   └── ...
└── apple_healthy/
    └── ...
```

> Recommended dataset: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 🔄 Training Flow

```
Dataset Folder
  ↓
image_dataset_from_directory (128×128, batch=32)
  ↓
Train / Val / Test Split
  ↓
cache() → shuffle() → prefetch(AUTOTUNE)
  ↓
RandomFlip + RandomRotation augmentation (train only)
  ↓
ResNet-50 / Custom CNN
  ↓
Adam(lr=1e-4) + categorical_crossentropy
  ↓
EarlyStopping(monitor=val_loss, patience=10,
              restore_best_weights=True)
  ↓
model.save("<crop>_model.keras")
```

---

## 📈 Key Callbacks

### EarlyStopping

All notebooks use:

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,   # always restores best checkpoint
    min_delta=0.0001,
)
```

> `restore_best_weights=True` ensures the saved model is always the best epoch, not the last.

---

## 🔮 Future Improvements

| Improvement | Impact |
|---|---|
| **Grad-CAM visualisation** | Show which leaf regions triggered the prediction |
| **Streamlit / FastAPI app** | Real-time disease diagnosis from uploaded photos |
| **ONNX export** | Edge deployment on mobile devices |
| **Multi-label classification** | Detect multiple diseases in one image |
| **Data augmentation with CLAHE** | Improve contrast for microscopy images |
| **LR scheduling** | Cosine annealing for faster convergence |
| **Model ensembling** | Average CNN + ResNet predictions for higher accuracy |

---

## 👤 Author

**Built by ANSH SINGH**  
Portfolio project demonstrating deep learning expertise in agricultural computer vision.
