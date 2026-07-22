"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - K-Means Clustering

This module accepts 2D coordinates from PCA, mathematically
partitions the strains into discrete clusters, and exports a colored scatter plot.
"""

import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import config


def run_kmeans_clustering(
    df_coords,
    k=4, # (k=4 based on 2024 literature)
    plot_name="pca_kmeans.png",
    plot_title="K-Means Clustering",
    xlabel="PC1",
    ylabel="PC2",
):
    """
    Accepts any 2D coordinate dataframe, assigns strains to k clusters,
    and generates a colored scatter plot. 
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

    # Fit K-Means centroid model directly on the reduced coordinate space to assign cluster IDs
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    coords_array = df_coords.values
    cluster_labels = kmeans.fit_predict(coords_array)

    # Append cluster assignment labels to a duplicate coordinate dataframe
    df_clusters = df_coords.copy()
    df_clusters["Cluster"] = cluster_labels

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        df_clusters.iloc[:, 0],  # X-axis (e.g., PC1)
        df_clusters.iloc[:, 1],  # Y-axis (e.g., PC2)
        c=df_clusters["Cluster"],
        cmap="Set1",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
    )

    plt.xlabel(xlabel, fontsize=11, weight="bold")
    plt.ylabel(ylabel, fontsize=11, weight="bold")
    plt.title(f"{plot_title} (k={k})", fontsize=14, weight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(
        *scatter.legend_elements(), title="Clusters", title_fontsize="11", fontsize="10"
    )
    plt.tight_layout()

    plt.savefig(os.path.join(config.OUTPUT_DIR, plot_name), dpi=300)
    plt.close()

    return df_clusters
