"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Biomarker Extractor

This module extracts the specific driver genes (biomarkers) that dictate
the separation of strains within both the linear PCA space and the
non-linear Autoencoder latent space.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def extract_pca_loadings(df, top_n=10):
    """
    Computes PCA loadings to identify the top linear driver genes.
    """
    print("Extracting linear driver genes via PCA loadings...")
    pca = PCA(n_components=2)
    pca.fit(df.values)

    # Loadings are the components
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
