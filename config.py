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
STAT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "stat_summary.csv")

# Visualization Outputs (Check these names carefully!)
HCA_DENDROGRAM_PLOT = os.path.join(OUTPUT_DIR, "hca_dendrogram.png")
PIE_CHART_PLOT = os.path.join(OUTPUT_DIR, "pangenome_pie_distribution.png")
PCA_PLOT = os.path.join(OUTPUT_DIR, "pca_clusters.png")
PCA_KMEANS_PLOT = os.path.join(OUTPUT_DIR, "pca_kmeans.png")
PCA_HCA_PLOT = os.path.join(OUTPUT_DIR, "pca_hca_clusters.png")

# Downstream Biomarker Outputs
PCA_LOADINGS_CSV = os.path.join(OUTPUT_DIR, "pca_driver_genes.csv")
PATHOGENICITY_RISK_CSV = os.path.join(OUTPUT_DIR, "pathogenicity_risk_profile.csv")
HCA_CLUSTERS_CSV = os.path.join(OUTPUT_DIR, "hca_clusters.csv")  