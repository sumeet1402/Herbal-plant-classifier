# 🌿 Herbal Plant Classifier

**An end-to-end computer-vision project that classifies plant images into 40 dataset categories using PyTorch, transfer learning, ResNet18, and Streamlit.**

[![Training workflow](https://github.com/sumeet1402/herbal-plant-classifier/actions/workflows/train-herbal.yml/badge.svg)](https://github.com/sumeet1402/herbal-plant-classifier/actions/workflows/train-herbal.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

# Herbal Plant Classifier

**An end-to-end computer-vision project that classifies plant images into 40 dataset categories using PyTorch, transfer learning, ResNet18, and Streamlit.**

<p align="center">
  <a href="https://herbal-plant-classifier-with-info-3n3ctc67frpfzijooubveq.streamlit.app/">
    <img src="https://img.shields.io/badge/TRY_THE_LIVE_APP-Launch_Classifier-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Try the Live Herbal Plant Classifier">
  </a>
</p>

<p align="center">
  <em>Upload a plant image and explore the classifier's predictions.</em>
</p>

<p align="center">
  <a href="https://github.com/sumeet1402/herbal-plant-classifier/actions/workflows/train-herbal.yml/badge.svg">
    <img src="https://github.com/sumeet1402/herbal-plant-classifier/actions/workflows/train-herbal.yml/badge.svg" alt="Training workflow">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch ResNet18">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
</p>

---

## Overview

This repository contains a reproducible image-classification pipeline based on the public [DIMPSAR Medicinal Plant Classification dataset](https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_plant_classification). The training workflow downloads the dataset, makes stratified train/validation/test splits, fine-tunes a pretrained ResNet18 network, evaluates the selected model on held-out images, publishes the model and reports as a GitHub Release, and commits evaluation files into `results/`.

> **Responsible use:** This is an educational image-classification demo, not a botanical authority or medical tool. Model predictions are not proof of plant identity, medicinal efficacy, toxicity, dosage, or safety. Do not use them to choose, prepare, or consume plants.

## Model

- **Architecture:** ResNet18 with ImageNet-pretrained weights
- **Framework:** PyTorch and TorchVision
- **Transfer learning:** freeze the earlier feature extractor; fine-tune the final residual block and classification head
- **Input size:** 224 × 224 RGB image
- **Loss:** cross-entropy
- **Optimizer:** AdamW
- **Class count:** inferred from the dataset labels (40 in the recorded dataset)
- **Data split:** stratified 70% training, 10% validation, 20% test
- **Random seed:** 42

The model checkpoint is selected using validation accuracy. Test metrics are calculated separately on the held-out test split.

## Evaluation

Every successful training run generates actual metrics and plots. The first successful run in this repository will populate the links below; do not report an accuracy number unless it is present in the generated metrics file.

- [Test metrics (`metrics.json`)](results/metrics.json)
- [Per-class precision, recall and F1 (`classification_report.csv`)](results/classification_report.csv)
- [Training history (`training_history.csv`)](results/training_history.csv)
- [Training and validation curves](results/training_curves.png)
- [Test confusion matrix](results/confusion_matrix.png)
- [Published model releases](https://github.com/sumeet1402/herbal-plant-classifier/releases)

## Automated training

The [GitHub Actions workflow](.github/workflows/train-herbal.yml) automatically runs when `train.py`, `app.py`, `requirements.txt`, or the workflow changes on `main`. It downloads data, trains and evaluates the classifier, uploads the checkpoint and reports to a GitHub Release, then commits the evaluation files to `results/`.

To start a run manually:
1. Open the **Actions** tab.
2. Select **Train Herbal Plant Classifier**.
3. Select **Run workflow** and choose 3, 5, or 8 epochs.

The default GitHub-hosted runner uses CPU training, which is slower than a GPU. You can inspect logs and results from the Actions run page.

## Run locally

Python 3.10 or 3.11 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies and train:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

Launch the interface after a trained model release exists:

```bash
streamlit run app.py
```

The app first checks for `artifacts/best_model.pt`; when absent, it downloads the latest checkpoint from this repository's public GitHub Release. The first successful training workflow must complete before a newly created repository can provide predictions.

Optional training environment variables:

- `HERBAL_EPOCHS` — epoch count (default 8)
- `HERBAL_BATCH_SIZE` — batch size (default 32)
- `HERBAL_LEARNING_RATE` — learning rate (default 0.0001)

## Project structure

```text
.
├── .github/workflows/train-herbal.yml  # automated training and publishing
├── .streamlit/config.toml              # theme settings
├── app.py                              # image-upload Streamlit app
├── train.py                            # dataset, training and evaluation pipeline
├── requirements.txt                    # dependencies
├── artifacts/                          # generated checkpoint (not in Git history)
└── results/                            # generated test metrics, report and plots
```

## Dataset and attribution

Dataset: [DIMPSAR Medicinal Plant Classification](https://huggingface.co/datasets/Project-AgML/DIMPSAR_medicinal_plant_classification), described by its publisher as 5,945 images across 40 classes under **CC BY 4.0**.

Citation provided by the dataset publisher: Pushpa, B. R., & Rani, N. Shobha (2023). *DIMPSAR: Dataset for Indian medicinal plant species analysis and recognition*. Data in Brief, 49, 109388.

Please preserve attribution and review the dataset license before redistributing source images or derivative assets.

## Interface implementation

The UI uses Python and native Streamlit components for layout, file upload, metrics, and predictions. A small Streamlit theme configuration controls the colors and typography; custom HTML is not required.

## License

The project code is provided under the MIT License. The dataset has its own CC BY 4.0 terms and attribution requirements.
