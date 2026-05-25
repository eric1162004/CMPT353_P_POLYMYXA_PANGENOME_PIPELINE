"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Configurationk Profile
Centralized paths, hyperparameters, and environemtn variables
"""

import os

# =====================================================================
# 1. DIRECTORY MATRIX CONFIGURATION
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Core Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Ensure output directory exists immediately upon import
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. STANDARDIZED FILE PATHS
# =====================================================================
# Data Pipeline Outputs
MATRIX_CSV = os.path.join(OUTPUT_DIR, "pangenome_matrix.csv")
STAT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "accessory_genome_heatmap.png")
PCA_PLOT = os.path.join(OUTPUT_DIR, "pca_clusters.png")
TSNE_PLOT = os.path.join(OUTPUT_DIR, "tsne_clusters.png")
NN_PLOT = os.path.join(OUTPUT_DIR, "nn_latent_space.png")

# Downstream Biomarker Outputs
NN_DRIVERS_CSV = os.path.join(OUTPUT_DIR, "nn_driver_genes.csv")
PCA_LOADINGS_CSV = os.path.join(OUTPUT_DIR, "pca_driver_genes.csv")
PATHOGENICITY_RISK_CSV = os.path.join(OUTPUT_DIR, "pathogenicity_risk_profile.csv")

# =====================================================================
# 3. ALGORITHM HYPERPARAMETERS
# =====================================================================
# K-Means Clustering Settings
K_MEAN_CLUSTERS = 2
RANDOM_STATE = 42

# t-SNE Settings
TSEN_PERPLEXITY = 2  # Scalable up to 5-30 for full 80-Strain cohort
TSNE_ITERATIONS = 1000

# Deep Learning (PyTorch Autoencoder) Settings
NN_EPOCHS = 100
NN_LEARNING_RATE = 0.01
NN_LATENT_DIM = 2
NN_HIDDEN_DIM = 256
