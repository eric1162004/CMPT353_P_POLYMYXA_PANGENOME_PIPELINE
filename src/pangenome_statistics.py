"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Pangenome Statistics

This module evaluates vertical presence/absence vectors across
the cohort, it stratifies genomic features into functional layers
based on frequency thresholds.

[ Cloud Layer ] ─> Rare/Unique Genes (< 15% presence)
[ Shell Layer ] ─> Adaptable/Shared Genes (15% - 95% presence)
[  Core Layer ] ─> Conserved Housekeeping Genes (≥ 95% presence)

Inputs:
    - config.MATRIX_CSV: Master binary presence/absence matrix (Strains x Genes).

Outputs:
    - config.STAT_SUMMARY_CSV: Tabular matrix detailing counts and percentages.
    - config.PIE_CHART_PLOT: High-contrast visualization of the genomic layout.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import config


def calculate_pangenome_stats(
    matrix_path=config.MATRIX_CSV, output_csv=config.STAT_SUMMARY_CSV
):
    """
    Loads the binary matrix, evaluates gene freqs, and classifies
    features into core, shell, and cloud genomic layers.
    """
    print(f"Loading binary matrix from: {matrix_path}")
    if not os.path.exists(matrix_path):
        print(f"Error: Matrix file missing.")
        return None

    # Load matrix (Strains = Rows, Genes = Columns)
    df = pd.read_csv(matrix_path, index_col=0)
    total_strains = len(df)
    total_genes = df.shape[1]

    print(f"Analyzing {total_genes} total unique genes across {total_strains} strains.")

    # 1. Calculate how many strains possess each gene (Column-wise sum)
    gene_counts = df.sum(axis=0)

    # 2. Calculate percentage presence per gene
    gene_percentages = gene_counts / total_strains * 100

    # 3. Classify based on standard bioinformatics freq thresholds
    # NOTE: Adding .index b/c we only care about the names of the
    # genes that passed the filter
    core_genes = gene_percentages[gene_percentages >= 95.0].index.to_list()

    # Shell sits between 15% and 95%
    shell_genes = gene_percentages[
        (gene_percentages >= 15.0) & (gene_percentages < 95.0)
    ].index.to_list()

    # Cloud represents rare or strain-specific elements
    cloud_genes = gene_percentages[gene_percentages < 15.0].index.to_list()

    # 4. Compile numerical summary
    stats_data = {
        "genomic_layer": ["Core", "Shell", "Cloud", "Total_Pangenome"],
        "gene_count": [
            len(core_genes),
            len(shell_genes),
            len(cloud_genes),
            total_genes,
        ],
        "percentage_of_total": [
            (len(core_genes) / total_genes) * 100,
            (len(shell_genes) / total_genes) * 100,
            (len(cloud_genes) / total_genes) * 100,
            100.0,
        ],
    }

    df_stats = pd.DataFrame(stats_data)
    df_stats.to_csv(output_csv, index=False)
    print(f"Statistical profile saved directly to: {output_csv}")

    return {
        "summary_table": df_stats,
        "core": core_genes,
        "shell": shell_genes,
        "cloud": cloud_genes,
    }


def plot_pangenome_distribution(df_stats, save_path=config.PIE_CHART_PLOT):
    """
    Generates a pie chart showing the distribution partition layout
    of the pangenome
    """
    # Isolate just our sub-components (rows 0,1,2)
    plot_df = df_stats[df_stats["genomic_layer"] != "Total_Pangenome"]

    labels = plot_df["genomic_layer"]
    counts = plot_df["gene_count"]

    # Plot settings
    colors = ["#2b5c8f", "#d97d24", "#4ca64c"]  # Core, Shell, Cloud
    explode = (0.05, 0.05, 0.05)

    plt.figure(figsize=(7, 7))
    plt.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=explode,
        textprops={"fontsize": 12, "weight": "bold"},
    )
    plt.title(
        "Paenibacillus polymyxa - Functional Pangenome Partitioning",
        fontsize=14,
        weight="bold",
        pad=20,
    )
    plt.tight_layout()

    # Export clear image file straight to disk
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Pie chart exported to: {save_path}\n")


def run_pangenome_analytics():
    """
    Calcualtes freq. and updates visualization portifolios
    """
    results = calculate_pangenome_stats()
    if results is not None:
        plot_pangenome_distribution(results["summary_table"])
    return results


if __name__ == "__main__":
    run_pangenome_analytics()
