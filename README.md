# Lung Cancer CT-Scan Classification

🔗 **[Live Demo](https://huggingface.co/spaces/Suizen/Pulmoscan-deploy)** · **[Deployment Repo](https://github.com/m6hhyrz78d-svg/PulmoScan)**

Comparative evaluation of four deep learning architectures — **VGG-16, EfficientNetV2, ConvNeXt, and Vision Transformer (ViT)** — for classifying lung CT-scan images into four classes (**Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, Normal**), with the best-performing model deployed as a live web app.

## Problem

Lung cancer is the leading cause of cancer death worldwide, and distinguishing between histological subtypes from CT imaging is difficult even for trained radiologists — the subtypes often look visually similar. This project asks: **which architecture generalizes best on a small (~1,000 image) medical imaging dataset**, and can the winning model be made usable by non-technical end users rather than staying a notebook?

## Approach

- Applied **transfer learning** with all four architectures pre-trained on ImageNet, backbone frozen, sharing an identical classification head (GAP → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Softmax)
- Trained under identical conditions (ReduceLROnPlateau, ModelCheckpoint, EarlyStopping) on Google Colab (T4 GPU, TensorFlow/Keras)
- Evaluated on a held-out test set using **weighted F1-score** as the primary metric (to account for class imbalance), alongside accuracy, precision, and recall
- Deployed the winning model via **FastAPI + Gradio**, hosted on **Hugging Face Spaces**, with a **GitHub Actions** CI/CD pipeline that auto-syncs every push to the live app

## Results

| Model | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| **VGG-16** | 0.619 | 0.68 | 0.62 | **0.619** |
| EfficientNetV2 | 0.619 | 0.75 | 0.62 | 0.556 |
| ViT | 0.587 | 0.59 | 0.59 | 0.557 |
| ConvNeXt | 0.575 | 0.59 | 0.57 | 0.504 |

**VGG-16 won** despite being the oldest/simplest architecture — its stronger inductive biases (locality, translation equivariance) made it more data-efficient than the transformer and modernized-CNN alternatives, which need more data to generalize. It also balanced precision and recall more evenly, which matters more than raw accuracy for a medical classifier.

## Tech Stack

`TensorFlow` / `Keras` · `PyTorch` / `Transformers` · `VGG-16` · `EfficientNetV2` · `ConvNeXt` · `ViT` · `OpenCV` · `scikit-learn` · `FastAPI` · `Gradio` · `Hugging Face Spaces` · `GitHub Actions` · Google Colab (T4 GPU)

## Setup

```bash
pip install -r requirements.txt
```

Developed and trained on Google Colab (T4 GPU).

## My Role

Led model development and evaluation, including architecture comparison methodology, and co-authored the accompanying research paper submitted to J-BEKEN.

---
*This work is part of an academic paper currently under submission to the J-BEKEN journal.*
