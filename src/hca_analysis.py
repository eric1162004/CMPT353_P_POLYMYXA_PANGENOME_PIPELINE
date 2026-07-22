"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - HCA Analysis

This module performs Hierarchical Agglomerative Clustering (HCA) on the binary
presence/absence matrix. Utilizing a Jaccard distance metric and complete linkage,
it mathematically partitions the strains into discrete evolutionary sub-lineages
and generates a dendrogram tree visualization.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import config


def plot_hca_dendrogram(df_matrix, save_path):
    """
    Computes the linkage matrix and plots the evolutionary branching dendrogram.
    """

    linkage_matrix = sch.linkage(df_matrix, method="complete", metric="jaccard")

    plt.figure(figsize=(12, 8))

    sch.dendrogram(
        linkage_matrix,
        labels=df_matrix.index.to_list(),
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.85,
    )

    plt.title(
        "P. polymyxa Pangenome - Hierarchical Clustering Dendrogram",
        fontsize=14,
        weight="bold",
        pad=15,
    )
    plt.xlabel("Strain ID", fontsize=11, weight="bold")
    plt.ylabel("Jaccard Distance", fontsize=11, weight="bold")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_hca_on_pca(df_pca, df_clusters, plot_path, xlabel="PC1", ylabel="PC2"):
    """
    Plots HCA cluster labels onto a PCA scatter plot with consistent styling.
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df_pca["PC1"],
        df_pca["PC2"],
        c=df_clusters["HCA_Cluster"],
        cmap="Set1",
        s=100,
        edgecolors="black",
        alpha=0.8,
    )
    plt.xlabel(xlabel, fontsize=11, weight="bold")
    plt.ylabel(ylabel, fontsize=11, weight="bold")
    plt.title("HCA Partitioning", fontsize=14, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300)
    plt.close()


def run_hca_clustering(
    matrix_path=config.MATRIX_CSV,
    n_clusters=4,
):
    """
    Loads the matrix, exports the dendrogram visualization, fits the HCA model,
    and exports the tabular cluster assignments.
    """
    if not os.path.exists(matrix_path):
        print("Error: Matrix file missing. Cannot run HCA.")
        return None

    df = pd.read_csv(matrix_path, index_col=0)

    plot_hca_dendrogram(df, config.HCA_DENDROGRAM_PLOT)

    hca = AgglomerativeClustering(
        n_clusters=n_clusters, metric="jaccard", linkage="complete"
    )

    cluster_labels = hca.fit_predict(df.values)

    df_clusters = pd.DataFrame({"HCA_Cluster": cluster_labels}, index=df.index)
    df_clusters.index.name = "Strain_ID"

    df_clusters.to_csv(config.HCA_CLUSTERS_CSV)

    return df_clusters


if __name__ == "__main__":
    run_hca_clustering()
