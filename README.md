# VAE for Hybrid Language Music Clustering

This repository implements an **unsupervised learning framework based on Variational Autoencoders (VAE)** for clustering **hybrid-language music** containing both **English and Bangla** songs.  
The project focuses on learning compact latent representations from combined **audio features** and **lyrics-based features**, and evaluating their effectiveness for clustering compared to a **PCA + K-Means baseline**.

The project was originally developed in **Google Colab** and later refactored into a modular, reproducible structure suitable for GitHub submission.

---

## Project Objectives

- Learn low-dimensional latent representations of music using a VAE
- Perform clustering in the learned latent space using K-Means
- Compare VAE-based clustering with a PCA-based baseline
- Analyze clustering quality using standard evaluation metrics
- Visualize latent space structure using t-SNE and UMAP

---

## Repository Structure

vae-music-clustering/
├── data/
│ ├── audio/ # placeholder (raw audio not included)
│ └── lyrics/ # placeholder (raw lyrics not included)
├── notebooks/
│ └── exploratory.ipynb # original Google Colab notebook
├── src/
│ ├── dataset.py # dataset generation and preprocessing
│ ├── vae.py # VAE model and loss function
│ ├── train.py # VAE training loop
│ ├── clustering.py # PCA, K-Means, and clustering evaluation
│ ├── visualization.py # t-SNE and UMAP visualizations
│ ├── analysis.py # cluster-level analysis utilities
│ └── main.py # main execution script
├── results/
│ ├── latent_visualization/
│ │ ├── tsne_vae.png
│ │ └── umap_vae.png
│ └── clustering_metrics.csv
├── requirements.txt
├── README.md

---

## Methodology Overview

### Feature Representation
- **Audio features**: Simulated MFCC-style vectors
- **Lyrics features**: TF-IDF embeddings from English and Bangla lyrics
- Final input features are formed by concatenating audio and lyrics representations

### Model
- Fully connected Variational Autoencoder (VAE)
- 8-dimensional latent space
- Reconstruction loss (MSE) + KL divergence regularization

### Clustering
- K-Means applied to VAE latent embeddings
- PCA + K-Means used as a baseline for comparison

### Evaluation Metrics
- Silhouette Score
- Calinski–Harabasz Index
- Language-based cluster purity

### Visualization
- t-SNE projection of latent space
- UMAP projection of latent space

---

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
