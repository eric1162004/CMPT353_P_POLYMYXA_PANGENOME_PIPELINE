# CMPT353_P_POLYMYXA_PANGENOME_PIPELINE (IN DEVELOPMENT)

An end-to-end computational biology and unsupervised machine learning pipeline designed to parse raw bacterial assembly data, engineer a global pangenomic presence/absence matrix, and characterize microbial lineages using comparative statistical and deep learning frameworks.

## 🧬 Project Overview
This platform bridges genomic engineering with unsupervised data science to study the functional diversity of *Paenibacillus polymyxa*. Because empirical phenotypic labels (such as plant pathogenicity vs. growth promotion) are only available for a small subset of sequenced strains, this pipeline leverages non-linear dimensional reduction and spatial clustering to map unknown genomes against characterized biological anchors.

---

## 📂 Repository Architecture

```text
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE/
│
├── config.py              # Centralized hyperparameters, paths, and environment settings
├── utils.py               # Shared utility functions (string formatting, attribute cleaning)
├── CONTROL_PANEL.ipynb    # Frontend Jupyter Notebook command dashboard
├── readme.md              # Project documentation and engineering specifications
│
├── data/                  # Flattened directory containing raw genomic strain folders
│   ├── GCF_001719045.1/
│   └── GCF_003956325.1/
│
├── src/                   # Single-responsibility backend modules
│   ├── __init__.py        # Packages the directory for clean system imports
│   ├── matrix_generator.py # Parses raw GFF3 annotation coordinates into a binary matrix
│   ├── pangenome_statistics.py # Computes core/shell/cloud divisions & accumulation curves
│   ├── pca_analysis.py    # Standardizes features and executes linear PCA projections
│   ├── kmeans_clustering.py # Performs spatial partitioning and cluster assignments
│   ├── tsne_embedding.py  # Maps high-dimensional neighbor probabilities using t-SNE
│   ├── pangenome_autoencoder.py # Trains a PyTorch Deep Autoencoder for latent space mapping
│   └── biomarker_extractor.py   # Extracts top driving features and runs risk factor profiles
│
└── outputs/               # Standardized pipeline artifacts and visualizations
    ├── pangenome_matrix.csv
    ├── pca_clusters.png
    ├── tsne_clusters.png
    ├── nn_latent_space.png
    └── nn_driver_genes.csv
└── references/...