"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Matrix Generator

Every single line in the GFF3 file that is marked as a "CDS" (Coding Sequence)
represents a specific gene or product that the strain physically possesses in
its genome. By looping through every single one of those rows, this script
builds a complete, unique genetic inventory for that specific strain.

GFF3 Rows (CDS features)                         Cleaned Strain Inventory
┌────────────────────────────────────────┐     ┌─────────────────────────┐
│ Row 45:  ...gene=paaA;product=kinase.. │ ──► │ - paaa                  │
│ Row 46:  ...gene=entE;product=ligase.. │ ──► │ - ente                  │
│ Row 47:  ...gene=gyrA;product=gyrase.. │ ──► │ - gyra                  │
└────────────────────────────────────────┘     └─────────────────────────┘
"""

import os
import glob
import re
import pandas as pd
import config


def clean_gene_name(name):
    """
    Standardizes messy genomic text strings.
    Removes special characters, converts to lowercase, and catches missing fields.
    Shared globally across data extraction and biomarker mining
    """
    if not isinstance(name, str) or pd.isna(name):
        return "hypothetical_protein"

    name = name.strip().lower()
    # Replace non-alphanumeric characters with underscores
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Collapse multiple consecutive underscores
    name = re.sub(r"_+", "_", name).strip("_")

    return name if name else "hypothetical_protein"


def parse_single_gff(gff_path):
    """
    Parses a single GFF3 file to extract, clean, and compile a unique set
    of functional coding sequences (CDS), filtering out structural RNAs and
    hypothetical proteins.
    """
    gene_set = set()

    with open(gff_path, "r") as file:
        for line in file:
            if line.startswith("#"):  # comment line
                continue

            parts = line.strip().split("\t")
            if len(parts) < 9:  # malformed line
                continue

            """
            NOTE: Isolate protein-coding sequences to focus specifically on metabolic 
            and virulence pathways while eliminating non-coding genomic noise.
            """
            if parts[2] == "CDS":
                attributes = parts[8]

                """
                Pattern:  gene=  (  [^;]  +  )
                            │    │    │   │  └─── Match this rule 1 or more times
                            │    │    │   └────── Group 1 boundary starts here
                            │    │    └────────── Rule: Any character that is NOT a semicolon
                            │    └─────────────── Capture boundary starts here
                            └──────────────────── Look for this literal text first
                """
                gene_match = re.search(r"gene=([^;]+)", attributes)
                product_match = re.search(r"product=([^;]+)", attributes)

                """
                NOTE:
                The gene= attribute contains the formal, internationally recognized gene symbol, 
                but there is no formal short symbol for it, it leaves the gene= attribute 
                completely blank or omits it.
                
                The product= string describes the literal protein function. By pulling from product= 
                when gene= is missing, the script ensures that recognized metabolic enzymes are 
                still captured
                
                .group(0) always returns the entire matched text
                .group(1) returns only the first explicit capture group
                """
                if gene_match:
                    raw_name = gene_match.group(1)
                elif product_match:
                    raw_name = product_match.group(1)
                else:
                    raw_name = "hypothetical_protein"  # unverified functional genes

                cleaned_name = clean_gene_name(raw_name)

                if cleaned_name != "hypothetical_protein":
                    gene_set.add(cleaned_name)

    return gene_set


def vectorize_pangenome(strain_gene_sets, global_pangenome_genes):
    """
    Converts individual strain gene sets into a standardized
    binary presence-absence Pandas DataFrame matrix.
    """
    matrix_rows = []
    strain_lists = []

    """
    strain_gene_sets =
    {
        "GCF_001719045.1": {"paaa", "ente", "gyra"},
        "GCF_003956325.1": {"ente", "gyra", "dnad"}
    }
    """
    for strain_id, genes_present in strain_gene_sets.items():
        strain_lists.append(strain_id)

        # 1 if present, 0 if absent
        row_vector = [
            1 if gene in genes_present else 0 for gene in global_pangenome_genes
        ]
        matrix_rows.append(row_vector)

    return pd.DataFrame(matrix_rows, index=strain_lists, columns=global_pangenome_genes)


def run_matrix_generation(data_dir=config.DATA_DIR, output_path=config.MATRIX_CSV):
    """
    Locates strain subdirectories, parses GFF3 annotations, computes the global
    pangenome feature union, and exports the resulting binary matrix to CSV.
    """
    # 1. Locate all individual strain subdir
    strain_folders = glob.glob(os.path.join(data_dir, "GCF_*"))
    if not strain_folders:
        print(f"Error: No strain folders starting with 'GCF_' found in {data_dir}")
        return None

    strain_gene_sets = {}

    # 2. Parse genomic data
    for folder in strain_folders:
        strain_id = os.path.basename(folder)  # ie. GCF_000146875.3
        gff_files = glob.glob(
            os.path.join(folder, "*.gff")
        )  # GFF (General Feature Format) file

        if not gff_files:
            print(f"Warning: No GFF file found in {folder}. Skipping.")
            continue

        # Call the parser unit
        strain_gene_sets[strain_id] = parse_single_gff(gff_files[0])

    # 3. Build master list of unique genes, sorted alphabetically
    """
    Take Union: {"paaa", "ente", "gyra"} U {"ente", "gyra", "dnad"} U ... 
    """
    global_panenome_genes = sorted(list(set.union(*strain_gene_sets.values())))

    # 4. Vectorize
    pangenome_matrix = vectorize_pangenome(strain_gene_sets, global_panenome_genes)

    # 5. Export results
    pangenome_matrix.to_csv(output_path)

    return pangenome_matrix


if __name__ == "__main__":
    run_matrix_generation()
