"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - K-Means Clustering

This module accepts 2D coordinates (from PCA or Autoencoder), mathematically
partitions the strains into discrete clusters, and exports a colored scatter plot.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import config


def run_kmeans_clustering(
    df_coords, k=4, plot_name="kmeans_clusters.png", plot_title="K-Means Clustering"
):
    """
    Accepts any 2D coordinate dataframe, assigns strains to k clusters,
    and generates a colored scatter plot. (k=4 based on 2024 literature)
    """
    if df_coords is None or df_coords.empty:
        print("Error: No coordinates provided to K-Means.")
        return None

    total_strains = len(df_coords)
    if total_strains < k:
        print(
            f"Warning: Only {total_strains} strains. Adjusting k down to {total_strains}."
        )
        k = total_strains

    print(f"Fitting K-Means model to discover {k} sub-lineages...")

    # 1. Fit K-Means directly on the provided coordinates
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    coords_array = df_coords.values
    cluster_labels = kmeans.fit_predict(coords_array)
    
    # 2. Add labels to a new DataFrame
    df_clusters = df_coords.copy()
    df_clusters["Cluster"] = cluster_labels

    # 3. Generate Colored Visualization
    plot_path = os.path.join(config.OUTPUT_DIR, plot_name)
    plt.figure(figsize=(10, 8))

    # Scatter plot using a colormap to separate clusters
    scatter = plt.scatter(
        df_clusters.iloc[:, 0],  # X-axis (e.g., PC1 or Latent Dim 1)
        df_clusters.iloc[:, 1],  # Y-axis (e.g., PC2 or Latent Dim 2)
        c=df_clusters["Cluster"],
        cmap="Set1",  # Distinct color map
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
    )

    x_label = df_clusters.columns[0]
    y_label = df_clusters.columns[1]

    plt.xlabel(x_label, fontsize=11, weight="bold")
    plt.ylabel(y_label, fontsize=11, weight="bold")
    plt.title(f"{plot_title} (k={k})", fontsize=14, weight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add a legend for the clusters
    plt.legend(
        *scatter.legend_elements(), title="Clusters", title_fontsize="11", fontsize="10"
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Colored K-Means cluster plot exported to: {plot_path}\n")

    return df_clusters
