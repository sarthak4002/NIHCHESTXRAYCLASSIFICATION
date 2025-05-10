# NIHCHESTXRAYCLASSIFICATION
AI-Driven Multi-Label Chest X-Ray Classification Detection Using Deep Learning
# 🫁 Chest X-ray Multi-Disease Classification

This project aims to develop a deep learning-based multi-label classification system to automatically detect multiple thoracic diseases from chest X-ray images. Using the NIH Chest X-ray dataset, the model predicts the presence of conditions such as "Pneumonia", "Pneumothorax", "Cardiomegaly", "Pleural_thickening", "Fibrosis", "No Finding". It includes interpretable visualizations like Grad-CAM and evaluates performance using robust metrics.

---

##  Features

-  Multi-label classification for up to 6 thoracic diseases
-  Multiple model architectures: ResNet18, EfficientNetB0/B2, MobileNet, Swin Transformer
-  Preprocessing with normalization and data augmentation
-  Evaluation using Accuracy, F1-Score, Precision, Recall, ROC-AUC
-  Explainability with Grad-CAM visualizations
-  Confusion matrices and ROC curves for analysis
-  Easy-to-run training, validation, and prediction scripts

---

## 📂 Dataset Used

- **NIH Chest X-ray14 Dataset**  
  - Source: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - 112,120 frontal-view chest X-ray images
  - 15 disease labels, used only 6 for this project:
"Pneumonia", "Pneumothorax", "Cardiomegaly", "Pleural_thickening", "Fibrosis", "No Finding"

---

## Model Architectures

- ResNet50
- EfficientNetB0
- EfficientNetB4
- MobileNetV2

All models are fine-tuned using PyTorch with customized classification heads for multi-label output.

---

##  Visualizations

![Screenshot 2025-05-04 195335](https://github.com/user-attachments/assets/5a07e724-0a1f-4e27-a6fa-2e29d662c351)

---
