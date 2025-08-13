# 🍅 Tomato Leaf Disease Detection with Attention-Augmented EfficientNet-B0

## 📌 Overview
This project implements a deep learning model for **tomato leaf disease classification** using the PlantVillage dataset.  
It extends the paper *"Feature Fusion-Based Hybrid Deep Learning Model for Tomato Disease Detection"* (Attallah, 2023) by replacing the original **multi-CNN + feature fusion + traditional classifier** pipeline with a **single, end-to-end EfficientNet-B0** enhanced with **Squeeze-and-Excitation (SE) attention**.

The result: **99.63% accuracy** with significantly lower complexity.

---

## ✨ Key Features
- **Modern Backbone:** EfficientNet-B0 pretrained on ImageNet.
- **Built-in Attention:** Squeeze-and-Excitation (SE) blocks for improved focus on disease-specific features.
- **End-to-End Training:** No manual feature fusion or selection.
- **High Accuracy:** Matches near state-of-the-art with fewer parameters.
- **Deployment-Ready:** Simple architecture for ONNX / TensorRT export.

---

## 📂 Dataset
- **Source:** [PlantVillage Tomato Subset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Classes:** 10 total — 9 diseased conditions + 1 healthy.
- **Preprocessing & Augmentation:**
  - Random resized crops
  - Horizontal flip
  - Color jitter
  - Normalization

---

## 🏗 Architecture
**Base:** EfficientNet-B0  
**Enhancements:**
1. Retained built-in SE channel attention.
2. Added an extra SE attention module after the final convolutional block.  
**Classifier Head:**  
`Global Average Pooling → Dropout (p=0.3) → Dense (1280 → 10)`

---

## ⚙️ Training Setup
| Parameter        | Value |
|------------------|-------|
| Loss Function    | CrossEntropyLoss |
| Optimizer        | Adam (LR = 1e-3) |
| Scheduler        | CosineAnnealingLR |
| Batch Size       | 64 |
| Epochs           | 20 |
| Hardware         | RTX 3060 Ti |
| Runtime          | ~1.5 hours |

---

## 📊 Results

**Validation Accuracy:** `99.63%`

**Detailed Classification Report:**

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Tomato__Tomato_YellowLeaf__Curl_Virus | 1.00 | 1.00 | 1.00 | 622 |
| Tomato__Tomato_mosaic_virus | 1.00 | 1.00 | 1.00 | 76 |
| Tomato__Target_Spot | 0.99 | 1.00 | 0.99 | 273 |
| Tomato_Spider_mites_Two_spotted_spider_mite | 1.00 | 0.99 | 0.99 | 332 |
| Tomato_Septoria_leaf_spot | 1.00 | 1.00 | 1.00 | 368 |
| Tomato_Leaf_Mold | 1.00 | 1.00 | 1.00 | 186 |
| Tomato_Late_blight | 1.00 | 0.99 | 0.99 | 388 |
| Tomato_healthy | 1.00 | 1.00 | 1.00 | 337 |
| Tomato_Early_blight | 0.98 | 1.00 | 0.99 | 210 |
| Tomato_Bacterial_spot | 1.00 | 1.00 | 1.00 | 411 |

**Overall:**
- **Accuracy:** 0.9963  
- **Macro Avg:** Precision 1.00 | Recall 1.00 | F1 1.00  
- **Weighted Avg:** Precision 1.00 | Recall 1.00 | F1 1.00  

---

## 📦 Installation
```bash
# Clone repo
git clone https://github.com/yourusername/tomato-disease-detection.git
cd tomato-disease-detection

# Install dependencies
pip install -r requirements.txt
