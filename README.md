# UO-Explainer: A Unified Approach to GNN Explanation Harnessing Human-Interpretable Structural Units

## 📌 Overview

This repository contains the official implementation of the UO-Explainer model, as introduced in our IEEE TNNLS submission.

## 🔧 Installation

To set up the environment, run:

```bash
conda env create -f environment.yml
conda activate kr
```

## 📊 Datasets

- **BAC (Barabási–Albert Community)**: Synthetic dataset based on Barabási–Albert model
- **BA Shapes (bashapes)**: BA graph with embedded shapes
- **Tree-Cycle (tree\_cycle)** & **Tree-Grid (tree\_grid)**: Synthetic graph datasets with structured topologies
- **PPI (ppi0-ppi5)**: Protein-Protein Interaction datasets

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

## 📜 Citation

If you use this code, please cite our IEEE TNNLS submission: 📄 **UO-Explainer: A Unified Approach to GNN Explanation Harnessing Human-Interpretable Structural Units**


