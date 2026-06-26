"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - t-SNE Analysis

This module performs t-Distributed Stochastic Neighbor Embedding (t-SNE)
on the pangenome matrix. It maps non-linear local relationships
between strains using a Jaccard distance metric.

NOTE: "embedding" refers to both the process of mapping high-dimensional
data into a lower-dimensional space (typically 2D or 3D) and the resulting
low-dimensional representation itself.

NOTE: Jaccard distance allows the algo to cluster sparse, categoricla, or genomic data
(such as microbial presence/absence) matrics.

NOTE: Preservation of Local Structure: If two points are very close to each other
in the high-dimensional space, the embedding forces them to be very close to
each other in the 2D plot.

Loss of Global Distance Meaning: The distance between widely separated clusters
in a t-SNE embedding does not accurately reflect their true distance or global
relationship in the high-dimensional space. Only local proximity is highly reliable.
"""

# NOTE: This code block is included to suppress warning from sklearn
# Scikit-learn tries to detect the system's physical CPU cores to
# optimize thread distribution. Calculate available logical threads.
# ------------------------------------------------------------------------
import os

total_cores = os.cpu_count()
os.environ["LOKY_MAX_CPU_COUNT"] = str(total_cores if total_cores else 4)
# ------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import config


def run_tsne_analysis(matrix_path=config.MATRIX_CSV, plot_path=config.TSNE_PLOT):
    """
    Loads the presence/absence matrix, compute a Jaccard distance matrix,
    fits a t-SNE embedding, and exports a 2D cluster visualization.
    """
    print(f"Loading matrix for t-SNE from: {matrix_path}")
    if not os.path.exists(matrix_path):
        print("Error: Matrix file missing. Cannot run t-SNE.")
        return None

    # Load  matrix (Stains = Rows, Genes = Columns)
    df = pd.read_csv(matrix_path, index_col=0)
    total_strains = len(df)

    # Check if we have enough samples to satisfy neighborhood constraints
    if total_strains < 2:
        print("Error: t-SNE requires at least 2 strains to run.")
        return None

    strain_ids = df.index.to_list()

    # 1. Compute Jaccard Distance Matrix
    # Because t-SNE defaults to Euclidean, we precompute the true binary
    # Jaccard distances to capture biological similarity accurately
    print("Calculating pairwise Jaccard distances...")
    binary_values = df.values.astype(bool)
    jaccard_distances = pairwise_distances(binary_values, metric="jaccard")

    # 2. Configure t-SNE Hyperparameters
    # Perplexity balances local vs global aspects of the data. With small
    # test cohorts, perplexity must be strictly less than the total number
    # of samples.
    default_perplexity = 30
    perplexity = min(default_perplexity, max(1, total_strains - 1))

    print(f"Fitting t-SNE embedding (Perplexity = {perplexity})")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        metric="precomputed",  # Tells sklearn we are feeding it a distance matrix
        init="random",
        random_state=42,  # Ensure identical projections across pipeline runs
        max_iter=1000,
    )

    X_tsne = tsne.fit_transform(jaccard_distances)

    # 3. Construct Coordinates DataFrame
    df_tsne = pd.DataFrame(X_tsne, columns=["t-SNE 1", "t-SNE 2"], index=strain_ids)

    # 4. Generate Visualization
    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_tsne["t-SNE 1"],
        df_tsne["t-SNE 2"],
        c="#2b5c8f",
        s=100,
        alpha=0.8,
        edgecolors="black",
        linewidths=1.5,
    )
    plt.title(
        "Pangenome Neighborhoods via Non-Linear t-SNE",
        fontsize=13,
        weight="bold",
        pad=15,
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=11, weight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=11, weight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()  # fixes the common issue where text overlays or clips out of view

    # Save figure
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"t-SNE cluster plot exported directly to: {plot_path}\n")

    return df_tsne


if __name__ == "__main__":
    run_tsne_analysis()
