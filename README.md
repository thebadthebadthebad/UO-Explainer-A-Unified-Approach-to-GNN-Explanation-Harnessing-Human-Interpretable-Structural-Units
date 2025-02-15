# UO-Explainer: A Unified Approach to GNN Explanation Harnessing Human-Interpretable Structural Units

📌 Overview

UO-Explainer is a novel Graph Neural Network (GNN) explanation framework that unifies various explanation techniques while leveraging human-interpretable structural units. This repository contains the official implementation of the UO-Explainer model, as introduced in our IEEE TNNLS submission.

🚀 Features

Supports multiple datasets: bac, bashapes, tree_cycle, tree_grid, ppi0-ppi5

Implements orbit basis learning and score learning for structured GNN explanations

Compatible with PyTorch Geometric (PyG) and optimized for CUDA 12.1

Includes pre-configured training scripts and hyperparameters for easy reproduction

🔧 Installation

To set up the environment, run:

conda env create -f environment.yml
conda activate kr

📊 Datasets

BAC (Barabási–Albert Community): Synthetic dataset based on Barabási–Albert model

BA Shapes (bashapes): BA graph with embedded shapes

Tree-Cycle (tree_cycle) & Tree-Grid (tree_grid): Synthetic graph datasets with structured topologies

PPI (ppi0-ppi5): Protein-Protein Interaction datasets

🏃 Usage

Train the UO-Explainer model using:

python run.py --dataset bac --o_epochs 5000 --s_epochs 500

For more options, check:

python run.py --help

📜 Citation

If you use this code, please cite our IEEE TNNLS submission: 📄 UO-Explainer: A Unified Approach to GNN Explanation Harnessing Human-Interpretable Structural Units

