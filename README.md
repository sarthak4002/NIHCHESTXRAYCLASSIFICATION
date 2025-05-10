# NIHCHESTXRAYCLASSIFICATION
AI-Driven Multi-Label Chest X-Ray Classification Detection Using Deep Learning
# 🫁 Chest X-ray Multi-Disease Classification

This project aims to develop a deep learning-based multi-label classification system to automatically detect multiple thoracic diseases from chest X-ray images. Using the NIH Chest X-ray dataset, the model predicts the presence of conditions such as Pneumonia, Tuberculosis, COVID-19, Lung Opacity, Lung Cancer, and 'No Findings'. It includes interpretable visualizations like Grad-CAM and evaluates performance using robust metrics.

---

## ✅ Features

- 📦 Multi-label classification for up to 6 thoracic diseases
- 🧠 Multiple model architectures: ResNet18, EfficientNetB0/B2, MobileNet, Swin Transformer
- 🛠️ Preprocessing with normalization and data augmentation
- 📊 Evaluation using Accuracy, F1-Score, Precision, Recall, ROC-AUC
- 🔍 Explainability with Grad-CAM visualizations
- 📈 Confusion matrices and ROC curves for analysis
- 💾 Easy-to-run training, validation, and prediction scripts

---

## 📂 Dataset Used

- **NIH Chest X-ray14 Dataset**  
  - Source: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - 112,120 frontal-view chest X-ray images
  - 15 disease labels, used only 6 for this project:
    - Pneumonia
    - Tuberculosis
    - COVID-19
    - Lung Opacity
    - Lung Cancer
    - No Findings

---

## 🧠 Model Architectures

- ResNet18
- EfficientNetB0
- EfficientNetB2
- MobileNetV2
- Swin Transformer (ViT-based)

All models are fine-tuned using PyTorch with customized classification heads for multi-label output.

---

## 📈 Performance Metrics

| Model            | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------------------|----------|-----------|--------|----------|---------|
| ResNet18         | 0.87     | 0.84      | 0.82   | 0.83     | 0.91    |
| EfficientNetB2   | 0.89     | 0.86      | 0.85   | 0.85     | 0.93    |
| Swin Transformer | 0.90     | 0.88      | 0.86   | 0.87     | 0.94    |

> *Note: These are sample values. Replace with actual evaluation results.*

---

## 📊 Visualizations

- ✅ **Grad-CAM Heatmaps**: Highlight regions of X-ray that influenced the model’s decisions
- ✅ **Confusion Matrix**: Multi-class confusion matrix for each disease label
- ✅ **ROC Curves**: One-vs-rest ROC curves for each class

![Grad-CAM Example](outputs/gradcam_example.png)
![Confusion Matrix](outputs/confusion_matrix.png)

---

## 🚀 How to Run

### 🔧 1. Clone the Repository
```bash
git clone https://github.com/yourusername/chest-xray-classifier.git
cd chest-xray-classifier
