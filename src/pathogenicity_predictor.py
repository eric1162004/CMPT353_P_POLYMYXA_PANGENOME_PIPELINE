"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Pathogenicity Predictor

This module scans the accessory genome profile of each strain for high-risk
functional terms (virulence, toxins, antibiotics resistance) and computes
a comparative pathogenicity risk index.
"""

import os
import pandas as pd
import numpy as np
import config


def evaluate_strain_risk(df_matrix):
    """
    Scans gene column labels for virulence/resistance keywords and computes
    a normalized risk index per strain.
    """
    print(
        "Scanning accessory genome for high-risk clinical and environmental factors..."
    )

    risk_keywords = [
        "toxin",
        "hemolysin",
        "virulence",
        "resistance",
        "antibiotic",
        "beta_lactamase",
        "bacteriocin",
        "penicillin_binding",
        "bacitracin",
    ]

    # Filter the pangenome columns for matches
    matched_genes = []
    for gene in df_matrix.columns:
        if any(keyword in gene.lower() for keyword in risk_keywords):
            matched_genes.append(gene)

    print(
        f"Identified {len(matched_genes)} potential risk-associated features in global pool."
    )

    if not matched_genes:
        print("No high-risk keyword signatures found in the current pangenome pool.")
        return pd.DataFrame(
            {
                "Risk_Associated_Gene_Count": 0,
                "Pathogenicity_Risk_Index": 0.0,
            },
            index=df_matrix.index,
        )

    # Isolate matrix to only risk features
    df_risk_matrix = df_matrix[matched_genes]

    # Calculate absolute count of risk features possessed by each strain (row sums)
    strain_risk_counts = df_risk_matrix.sum(axis=1, numeric_only=True)

    # Normalize score between 0.0 to 1.0 based on max possible detected risk genes
    max_detected = int(strain_risk_counts.max())
    max_detected = max_detected if max_detected > 0 else 1
    
    risk_index = strain_risk_counts / max_detected

    df_profiles = pd.DataFrame(
        {
            "Risk_Associated_Gene_Count": strain_risk_counts,
            "Pathogenicity_Risk_Index": np.round(risk_index, 3),
        },
        index=df_matrix.index
    )

    return df_profiles.sort_values(by="Pathogenicity_Risk_Index", ascending=False)


def run_pathogenicity_prediction(
    matrix_path=config.MATRIX_CSV, output_path=config.PATHOGENICITY_RISK_CSV
):
    """
    Main execution pathway for risk profiling
    """
    print("Initializing Pathogenicity Predictor Engine..")
    if not os.path.exists(matrix_path):
        print("Error: Master matrix missing. Cannot compute risk profiles.")
        return None

    df_matrix = pd.read_csv(matrix_path, index_col=0)

    # Compute profiles
    df_risk_profiles = evaluate_strain_risk(df_matrix)

    # Export results
    df_risk_profiles.to_csv(output_path)
    print(f"Pathogenicity risk profile records exported to: {output_path}\n")

    return df_risk_profiles


if __name__ == "__main__":
    run_pathogenicity_prediction()