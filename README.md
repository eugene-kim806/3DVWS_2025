# 🧠 3D Auto-Segmentation of Post-Treatment Gliomas on mpMRI (BraTS 2024)

[![License: Academic Use](https://img.shields.io/badge/license-Academic--Only-lightgrey.svg)](https://www.med.upenn.edu/cbica/brats2024/data.html)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-In%20Progress-yellow.svg)]()

> 3D Vision Winter School 2025 – Individual Project

---

## 📚 Table of Contents

- [📌 Project Overview](#-project-overview)
- [🗂️ Dataset](#️-dataset)
- [🛠️ Features](#️-features)
- [🏗️ Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [📊 Results](#-results)
- [📈 Visualizations](#-visualizations)
- [🔮 Future Work](#-future-work)
- [🧑‍💻 Author](#-author)
- [📄 License](#-license)

---

## 📌 Project Overview

This project focuses on **automatic 3D segmentation of post-treatment gliomas** using multiparametric MRI (mpMRI) from the **BraTS 2024 dataset**. The goal is to segment clinically relevant tumor sub-regions:

- 🔴 Non-enhancing tumor core (NETC)
- 🟢 Surrounding non-enhancing FLAIR hyperintensity (SNFH)
- 🔵 Enhancing tissue (ET)
- 🟡 Resection cavity (RC)

  <img src="materials/GT_whole_rotate3D.gif" width="400"/>
  <img src="materials/labels_2D.png" width="500"/>

Developed as part of my individual project at the 3D Vision Winter School 2025, this repository includes preprocessing, training, inference, evaluation, and visualization modules.

---

## 🗂️ Dataset

The project uses data from the **BraTS 2024 Adult Glioma Post-Treatment Challenge**. Each subject contains:

- 4 mpMRI sequences: `t1n`, `t1c`, `t2w`, `t2f`
  
  <img src="materials/t1n_rotate3D.gif" width="350"/>
  <img src="materials/t1c_rotate3D.gif" width="350"/>
  <img src="materials/t2w_rotate3D.gif" width="350"/>
  <img src="materials/t2f_rotate3D.gif" width="350"/>
- 1 segmentation label map (`seg.nii.gz`)
  
  <img src="materials/example_seg_rotate3D.gif" width="350"/>
  
Label mapping:
- 0 = Background
- 1 = Non-enhancing Tumor Core (NETC)
- 2 = SNFH
- 3 = Enhancing Tumor (ET)
- 4 = Resection Cavity (RC)

> 🔒 Validation labels were not provided due to challenge policy. Inference results were evaluated qualitatively or via comparison to BraTS 2024 baseline algorithms (provided).

---

## 🛠️ Features

✅ Modular structure (exploration → preprocessing → training → evaluation)  
✅ Patch-wise 3D U-Net for memory-efficient segmentation  
✅ Supports 5-fold cross-validation  
✅ Lesion-wise Dice Score & Hausdorff 95 computation  
✅ 3D Slicer-compatible output for visualization  
✅ Inference comparison with BraTS 2024 baseline model

---

## 🏗️ Project Structure

```

3DVWS_2025/
│
├── data/                      
│   ├── raw/                   # Raw .nii.gz files per subject
│   └── split_folds/          # Split folds (.csv)
│
├── src/
│   ├── dataset/               # Custom PyTorch dataset & loader
│   ├── preprocessing/         # Resampling, normalization
│   ├── models/                # 3D U-Net and other architectures
│   ├── training/              # Training loop, logging
│   ├── evaluation/            # Metric computation
│   └── visualization/         # 3D visualization helpers
│
├── results/                   # Inference masks, visualizations
├── requirements.txt
└── README.md

```

---

## 🚀 Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/eugene-kim806/3DVWS_2025.git
cd 3DVWS_2025
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Organize the data

Place BraTS 2024 data under the `data/raw/TrainingData` directory:

```
data/raw/TrainingData/BraTS-GLI-00005-100/
├── BraTS-GLI-00005-100-t1c.nii.gz
├── BraTS-GLI-00005-100-t1n.nii.gz
├── BraTS-GLI-00005-100-t2f.nii.gz
├── BraTS-GLI-00005-100-t2w.nii.gz
└── BraTS-GLI-00005-100-seg.nii.gz
```

### 4. Run the main script

```bash
python main.py
```

---

## 📊 Results

Table comparing dice scores and Hausdorff distance at 95th percentile between the challenge winner’s model and 3D U-Net model for each lesion type.

| Fold | Epochs | Model        | Dice ET | Dice NETC | Dice SNFH | Dice RC |   HD95 ET   |   HD95 NETC   |   HD95 SNFH   |   HD95 RC   |
|------|--------|--------------|---------|-----------|-----------|---------|-------------|---------------|---------------|-------------|
| 1    | 10     | 3D U-Net     | 0.9248  |  0.0112   |  0.9474   | 0.8495  | 1.4142 mm   |   31.064 mm   |   1.4142 mm   |   5.0990 mm |
| "    | "      | BraTS Winner |     0.9738    |      0.0905     |   0.9831        |     0.9378    |     1.0000 mm    |     39.810 mm      |     1.0000 mm      |     1.0000 mm    |

> ⚠️ Due to hardware and time limitations, only Fold 1 was trained for 10 epochs.

---

## 📈 Visualizations

Visual comparison of predicted segmentations using:

* **3D Slicer** overlays

---

## 🔮 Future Work

* ✅ Train full 5 folds
* ✅ Implement AMP, gradient accumulation, checkpointing
* ✅ Deploy interactive 3D demo using Gradio
* ✅ Compare lesion-level predictions across models
* ✅ Optimize post-processing for label consistency

---

## 🧑‍💻 Author

**\[Eugene Kim]**

Winter School on 3D Vision 2025

📧 \[[eugekim.Kim@ghent.ac.kr](mailto:eugekim.Kim@ghent.ac.kr)]

🔗 \[www.linkedin.com/in/eugene-kim-bb7544304]

🐙 \[github.com/eugene-kim806]

---

## 📄 License

This project is for academic use only. Please refer to the BraTS License Agreement for dataset usage.
