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

# Visualization Outputs (Check these names carefully!)
PIE_CHART_PLOT = os.path.join(OUTPUT_DIR, "pangenome_pie_distribution.png")
HEATMAP_PLOT = os.path.join(OUTPUT_DIR, "accessory_genome_heatmap.png")
PCA_PLOT = os.path.join(OUTPUT_DIR, "pca_clusters.png")
PCA_KMEANS_PLOT = os.path.join(OUTPUT_DIR, "pca_kmeans.png")

# Downstream Biomarker Outputs
PCA_LOADINGS_CSV = os.path.join(OUTPUT_DIR, "pca_driver_genes.csv")
PATHOGENICITY_RISK_CSV = os.path.join(OUTPUT_DIR, "pathogenicity_risk_profile.csv")

# =====================================================================
# 3. ALGORITHM HYPERPARAMETERS
# =====================================================================
# K-Means Clustering Settings
K_MEAN_CLUSTERS = 2
RANDOM_STATE = 42