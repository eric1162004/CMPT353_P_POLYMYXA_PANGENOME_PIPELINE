"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Biomarker Extractor

This module extracts the specific driver genes (biomarkers) that dictate
the seperation of strains within both the linear PCA space and the
non-linear Autoencoder latent space
"""

import os
from statistics import correlation
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import config


def extract_pca_loadings(df, top_n=10):
    """
    Computes PCA loadings to identify the top linear driver genes.
    """
    print("Extracting linear driver genes via PCA loadings...")
    pca = PCA(n_components=2)
    pca.fit(df.values)

    # Loadings are the components multiplied  by the squres root of the eigenvalues
    loadings_pc1 = pca.components_[0]
    loadings_pc2 = pca.components_[1]

    df_loadings = pd.DataFrame(
        {
            "Gene_Name": df.columns,
            "PC1_Loading": loadings_pc1,
            "PC2_Loading": loadings_pc2,
            "Absolute_PC1_Impact": np.abs(loadings_pc1),
            "Absolute_PC2_Impact": np.abs(loadings_pc2),
        }
    )

    # Sort by absolute impact to find top drivers
    top_pc1 = df_loadings.sort_values(by="Absolute_PC1_Impact", ascending=False).head(
        top_n
    )
    top_pc2 = df_loadings.sort_values(by="Absolute_PC2_Impact", ascending=False).head(
        top_n
    )

    return top_pc1, top_pc2


def extract_nn_latent_drivers(df_matrix, df_latent, top_n=10):
    """
    Identifies non-linear drivers by calculating the point-biserial correlation
    between binary gene prsense/absence and continuous latent space coordinates.
    """
    print("Mining non-linear drivers via latent space correlation profiling...")
    gene_names = df_matrix.columns
    latent_dims = df_latent.columns

    correlation_results = []

    # Calculate how strongly each gene's presence/absence shifts the latent coords
    for gene in gene_names:
        gene_vector = df_matrix[gene].values

        # Skip completely invariant genes if they bypassed previous filters
        if np.std(gene_vector) == 0:
            continue

        corrs = []
        for dim in latent_dims:
            latent_vector = df_latent[dim].values
            # Compute Pearson correlation between binary array and continuous array
            corr = np.corrcoef(gene_vector, latent_vector)[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0.0)

        correlation_results.append(
            {
                "Gene_Name": gene,
                "Latent_Dim1_Corr": corrs[0],
                "Latent_Dim2_Corr": corrs[1],
                "Max_Absolute_Correlation": max(abs(corrs[0]), abs(corrs[1])),
            }
        )

    df_nn_drivers = pd.DataFrame(correlation_results)
    top_nn_drivers = df_nn_drivers.sort_values(
        by="Max_Absolute_Correlation", ascending=False
    ).head(top_n)

    return top_nn_drivers


def run_biomarker_extraction(
    matrix_path=config.MATRIX_CSV,
    latent_path=config.EMBEDDING_CSV,
    pca_out=config.PCA_LOADINGS_CSV,
    nn_out=config.NN_DRIVERS_CSV,
):
    """
    Main execution pipeline for biomarker feature mining.
    """
    print(f"Initializing Biomarker Extractor Engine...")
    if not os.path.exists(matrix_path) or not os.path.exists(latent_path):
        print("Error: Missing required upstream matrix or latent coordinate files.")
        return None

    # Load datasets
    df_matrix = pd.read_csv(matrix_path, index_col=0)
    df_latent = pd.read_csv(latent_path, index_col=0)

    # 1. Run PCA Extraction
    top_pc1, top_pc2 = extract_pca_loadings(df_matrix, top_n=15)
    df_pca_drivers = pd.concat([top_pc1, top_pc2]).drop_duplicates(subset=["Gene_Name"])
    df_pca_drivers.to_csv(pca_out, index=False)
    print(f"Top PCA linear drivers saved directly to: {pca_out}")

    # 2. Run Latent Correlation Extraction
    df_nn_drivers = extract_nn_latent_drivers(df_matrix, df_latent, top_n=15)
    df_nn_drivers.to_csv(nn_out, index=False)
    print(f"Top Neural Network non-linear driver saved directly to: {nn_out}\n")

    return df_pca_drivers, df_nn_drivers


if __name__ == "__main__":
    run_biomarker_extraction()
