# UO-Explainer: A Unified Approach to GNN Explanation Harnessing Human-Interpretable Structural Units

## 📌 Overview

This repository contains the official implementation of the UO-Explainer model, as introduced in our IEEE TNNLS submission.

## 🔧 Installation

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate UOExplainer
```

## 📊 Datasets
Lastfm and gene datasets: Please refer to the paper's references for details.

- **BAC (Barabási–Albert Community)**
- **BA Shapes (bashapes)**
- **Tree-Cycle (tree\_cycle)** & **Tree-Grid (tree\_grid)**
- **PPI (ppi0-ppi5)**


## 🏃 Usage

Train the UO-Explainer model using:

```bash
cd UOExplainer
python run.py --dataset bac --o_epochs 5000 --s_epochs 500
```

For more options, check:

```bash
python run.py --help
```

