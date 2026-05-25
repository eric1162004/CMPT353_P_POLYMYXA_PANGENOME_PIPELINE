"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Accessory Genome Heatmap

This module generates a hierarchically clustered heatmap (clustermap)
of the accessory genome. It isolates variable traits by dropping converved
core features, revealing genomic sub-lineages.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import config


def run_heatmap_analysis(matrix_path=config.MATRIX_CSV, plot_path=config.HEATMAP_PLOT):
    """
    Extracts the accessory genome, performs hierarchical clustering,
    and exports a cluster map
    """
    print(f"Loading matrix for Heatmap from: {matrix_path}")
    if not os.path.exists(matrix_path):
        print("Error: Matrix file missing. Cannot run Heatmap")
        return None

    # Load matrix (Strains = Rows, Genes = Columns)
    df = pd.read_csv(matrix_path, index_col=0)
    total_strains = len(df)

    if total_strains < 2:
        print("Error: Heatmap clustering requires at least 2 strains.")
        return None

    # 1. Isolate the Accessory Genome (Drop the core genes)
    # Calculate strain-prevalence per gene
    gene_counts = df.sum(axis=0)
    gene_percentages = (gene_counts / total_strains) * 100

    # Keep only genes present in < 95% of strains
    accessory_genes = gene_percentages[gene_percentages < 95.0].index.to_list()
    df_accessory = df[accessory_genes]

    print(
        f"Core filtered out. Plotting {df_accessory.shape[1]} variable accessory gene."
    )

    if df_accessory.shape[1] == 0:
        print(
            "Warning: No accessory genes found with current thresholds. Skipping plot."
        )
        return None

    # 2. Configure plotting
    cmap = sns.color_palette(["#f2f2f2", "#2b5c8f"])  # absense = white, presence = blue

    # Determine sizing adjustments based on feature scale
    # If the gene pool is still wide, we compress the x-axis labels
    show_gene_labels = df_accessory.shape[1] <= 50
    
    # pre-computing row_linkage with SciPy 
    print("Computing stable hierarchical linkage trees...")
    row_linkage = sch.linkage(df_accessory, method="average", metric="jaccard")

    g = sns.clustermap(
        df_accessory,
        row_linkage=row_linkage, 
        #col_linkage=col_linkage,
        col_cluster=False,
        cmap=cmap,
        figsize=(10, 8),
        cbar_kws={"label": "Gene Presence (1: Blue, 0: Gray)", "ticks": [0, 1]},
        xticklabels=show_gene_labels,  # hide gene labels if too dense
        yticklabels=True,
        linewidths=0.1 if df_accessory.shape[1] <= 100 else 0.0,
    )

    # Clean up angle rotations for reading
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, weight="bold", fontsize=9)
    if show_gene_labels:
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)

    g.fig.suptitle(
        "Accessory Genome Hierarchical Clustering Map",
        fontsize=14,
        weight="bold",
        y=1.02,
    )

    # 3. Save figure to disk
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hierarchical clustermap exported straight to: {plot_path}\n")

    return df_accessory


if __name__ == "__main__":
    run_heatmap_analysis()
