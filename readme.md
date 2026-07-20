# Unsupervised Pangenomic Analysis of *Paenibacillus polymyxa*

> **Bioinformatic Pipeline for Genomic Architecture & Risk-Based Functional Profiling**

---

## 🔬 Overview

This bioinformatic data science pipeline processes raw genomic draft assemblies into a structured mathematical format to perform unsupervised machine learning on ***Paenibacillus polymyxa*** strains. It addresses the fundamental challenge of resolving ambiguous species boundaries, evaluating evolutionary sub-lineages, and automating risk-based functional profiling in the absence of empirical pathogenicity data.

---

## 🛠️ Required Libraries

To execute this pipeline, ensure your Python environment has the following dependencies installed:

*   **`pandas`** — for matrix compilation and tabular manipulation
*   **`numpy`** — for mathematical operations
*   **`matplotlib`** — for visualizations, scatter plots, and pie charts
*   **`scikit-learn`** — for implementing PCA, K-Means, and Agglomerative Clustering algorithms
*   **`scipy`** — for hierarchical clustering linkage matrices
*   **`jupyter` / `IPython`** — to render and execute the control panel notebook

---

## 📂 Repository Structure

The codebase and file architecture are organized as follows:

```text
├── CONTROL_PENEL.ipynb    # Primary interactive Jupyter Notebook for end-to-end execution
├── config.py              # Centralized configuration script for file paths & global variables
├── data/                  # Input directory (RefSeq assemblies: GCF_... / .fna / genomic.gff)
├── src/                   # Modular backend Python scripts
│   ├── matrix_generator.py
│   ├── pangenome_statistics.py
│   ├── pca_analysis.py
│   ├── hca_analysis.py
│   ├── kmeans_clustering.py
│   └── functional_profiler.py
└── outputs/               # Destination directory for generated CSV datasets & PNG plots
```

---

## 🚀 How to Run the Code

The recommended method for executing the project is via the interactive control panel:

1. Ensure the raw ***P. polymyxa*** GFF3 files are correctly placed within the `data/` directory.
2. Open `CONTROL_PENEL.ipynb` in your Jupyter environment.
3. Run the cells sequentially. The notebook will automatically import the modules from `src/` to:
   * **Generate the Matrix:** A custom parser extracts Coding Sequences (CDS) from the GFF3 files, explicitly filtering out hypothetical proteins, and builds a binary presence/absence matrix.
   * **Stratify the Pangenome:** Genes are grouped by frequency into **Core** ($\ge 95\%$), **Shell** ($15	ext{--}95\%$), and **Cloud** ($<15\%$) layers.
   * **Extract Drivers & Cluster:** Performs Principal Component Analysis (PCA) to extract top gene-loading drivers, followed by partitioning the matrix using K-Means ($k=4$) and Hierarchical Cluster Analysis (HCA) using a Jaccard distance metric.
   * **Risk Screening:** Scans the gene column labels for target functional keywords (e.g., toxin, resistance, bacteriocin) to assign a normalized Pathogenicity Risk Index.

---

## 📊 Output Files

Executing the pipeline will populate the `outputs/` folder with the following key artifacts:

*   `pangenome_matrix.csv` — The cleaned $62X4460$ binary feature matrix.
*   `stat_summary.csv` & `pangenome_pie_distribution.png` — Tabular and visual breakdowns of the pangenome architecture.
*   `pca_kmeans.png` & `pca_hca_clusters.png` — Side-by-side scatter plots mapping structural variance and subpopulation assignments.
*   `hca_dendrogram.png` — The complete linkage dendrogram mapping evolutionary branching.
*   `pathogenicity_risk_profile.csv` — The final dataset combining cluster assignments, target-associated gene counts, and the normalized Target Functional Index for all strains.

---

## 📥 Data Acquisition

To obtain the exact 62 ***Paenibacillus polymyxa*** strains analyzed in the Maggi et al. (2024) study, follow these steps:

1. Download the supplementary Excel spreadsheet, **File S4** (`AEM01740-24-s0004.xlsx`), which contains the finalized list of RefSeq assembly accession numbers that successfully passed the researchers' rigorous quality control filtering.
2. Extract these 62 specific accession numbers from the spreadsheet.
3. Automate the mass retrieval process by feeding them into the NCBI datasets command-line tool (`ncbi-datasets-cli`). By executing a batch command with this tool, you can instruct the NCBI database to download all the requested high-quality FASTA genome sequences alongside their precise GFF3 annotation maps simultaneously. 
4. These will arrive bundled in a single compressed ZIP file, ready to be extracted directly into your pipeline's `data/` directory.

---

## 🎓 CMPT 353 Grading Note

As required by the course guidelines, the instructor and TAs (**ggbaker**, **yha281**, **fma44**, **spa176**) have been added as collaborators to this Git repository for code evaluation.