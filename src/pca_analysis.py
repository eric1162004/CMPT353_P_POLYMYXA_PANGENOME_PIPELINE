"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - PCA Analysis

This module performs Principal Component Analysis (PCA) on the binary
presense/abserce matrix. It reduces the sparse, high-dimensional gene
space into an interpretable 2D embedding space to uncover structural
clusters among the strains.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config


def run_pca_analysis(matrix_path=config.MATRIX_CSV, plot_path=config.PCA_PLOT):
    """
    Loads the presence/absence matrix, fits a 2-component PCA model,
    and exports a scatter plot of the genomic variations.
    """
    print(f"Loading matrix for PCA from: {matrix_path}")
    if not os.path.exists(matrix_path):
        print("Error: Matrix file missing. Cannot run PCA.")
        return None

    # Load Matrix
    df = pd.read_csv(matrix_path, index_col=0)

    # Check if we have enough samples to run PCA
    if len(df) < 2:
        print("Error: PCA requires at least 2 strains to run.")
        return None

    # Extract sample IDs (strains) and binary vectors
    strain_ids = df.index.to_list()
    X = df.values

    # Initialize and fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Capture variance explain
    var_exp = pca.explained_variance_ratio_ * 100
    print(
        f"PCA complete. PC1 explains {var_exp[0]:.1f}%, PC2 explains {var_exp[1]:.1f}%"
    )

    # Construct coordinates DataFrame
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=strain_ids)

    # Generate the scientific visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_pca["PC1"],
        df_pca["PC2"],
        c="#2b5c8f",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
    )

    # Annotate points with their strain IDs
    for strain, row in df_pca.iterrows():
        plt.annotate(
            strain,
            (row["PC1"], row["PC2"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            weight="bold",
        )

    # Save figure
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"PCA cluster plot exported directly to: {plot_path}\n")

    return df_pca


if __name__ == "__main__":
    run_pca_analysis()
