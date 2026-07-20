"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - PCA Analyzer

This module encapsulates principal component analysis state and execution logic,
performing dimensionality reduction on the binary pangenome feature matrix,
calculating explained variance ratios, extracting top gene loading drivers,
and serializing scatter plot visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import config


class PangenomePCA:
    """
    Encapsulates PCA state to ensure loadings are always
    derived from the correct fitted model.
    """

    def __init__(self):
        self.model = None
        self.feature_names = None

    def run_analysis(
        self,
        matrix_path=config.MATRIX_CSV,
        plot_path=config.PCA_PLOT,
        cluster_labels=None,
    ):
        if not os.path.exists(matrix_path):
            print(f"Error: Matrix file missing at {matrix_path}")
            return None

        df = pd.read_csv(matrix_path, index_col=0)
        self.feature_names = df.columns.to_list()

        # Fit principal component analysis model to project binary features into 2D space
        self.model = PCA(n_components=2)
        X_pca = self.model.fit_transform(df.values)

        # Calculate percentage of total variance explained by each principal component
        variance = self.model.explained_variance_ratio_ * 100

        # Generate and save the PCA scatter plot visualization
        self._generate_plot(X_pca, plot_path, cluster_labels, variance)

        return pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df.index)

    def _generate_plot(self, X_pca, plot_path, cluster_labels, variance):
        plt.figure(figsize=(10, 8))

        # Render scatter plot coordinates with optional cluster coloring styles
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=cluster_labels or "#2b5c8f",
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidths=1.5,
        )

        # Dynamic Labels
        plt.title(
            "P. polymyxa Pangenome Structural Variance (PCA)",
            fontsize=14,
            weight="bold",
            pad=15,
        )
        plt.xlabel(f"PC1 ({variance[0]:.1f}%)", fontsize=11, weight="bold")
        plt.ylabel(f"PC2 ({variance[1]:.1f}%)", fontsize=11, weight="bold")

        plt.grid(True, linestyle="--", alpha=0.5)

        if cluster_labels is not None:
            plt.legend(*scatter.legend_elements(), title="Clusters")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def extract_loadings(self, feature_names, top_n=10):
        if self.model is None:
            raise RuntimeError("Run analysis() before extracting loadings.")

        # Extract underlying coordinate weight vectors for primary components from fitted model
        loadings_pc1 = self.model.components_[0]
        loadings_pc2 = self.model.components_[1]

        df_loadings = pd.DataFrame(
            {
                "Gene_Name": feature_names,
                "PC1_Loading": loadings_pc1,
                "PC2_Loading": loadings_pc2,
                "Abs_PC1": np.abs(loadings_pc1),
                "Abs_PC2": np.abs(loadings_pc2),
            }
        )

        return (
            df_loadings.sort_values("Abs_PC1", ascending=False).head(top_n),
            df_loadings.sort_values("Abs_PC2", ascending=False).head(top_n),
        )

    def get_x_label(self):
        """Returns the formatted label for the PC1 axis."""
        if self.model is None:
            return "PC1"
        variance = self.model.explained_variance_ratio_ * 100
        return f"PC1 ({variance[0]:.1f}%)"

    def get_y_label(self):
        """Returns the formatted label for the PC2 axis."""
        if self.model is None:
            return "PC2"
        variance = self.model.explained_variance_ratio_ * 100
        return f"PC2 ({variance[1]:.1f}%)"
