"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Target Functional Profiler

OVERVIEW:
    This module analyzes a pangenome matrix to quantify the relative functional
    abundance and metabolic potential of individual strains based on a user-defined
    vocabulary of target keywords. It identifies specific functional features
    (e.g., secondary metabolites, resistance determinants, or metabolic pathways)
    and generates a comparative prioritization profile for each strain.

ALGORITHM DESCRIPTION:
    1. FILTERING:
       The module performs a case-insensitive regex search across all gene columns
       using a provided set of target keywords. Columns containing any of these
       keywords are identified as 'target-associated features.'

    2. RAW COUNT CALCULATION:
       For each strain (row), the algorithm sums the presence (binary values)
       all identified target-associated feature columns. This yields a raw
       count representing the absolute number of matching functional factors
       identified per strain.

    3. NORMALIZATION (Relative Abundance / Priority Index):
       To provide a standardized comparison, the raw counts are normalized.
       The index is calculated using the following formula:

            Target_Index = Strain_Raw_Count / Max_Detected_Count_In_Dataset

       - The strain(s) with the highest number of matching genes serves as
         the benchmark (1.0).
       - All other strains are ranked relative to this maximum, resulting in a
         value between 0.0 and 1.0.

    4. OUTPUT:
       Results are generated as a DataFrame containing the raw target gene count
       and the normalized index, sorted in descending order of priority for
       immediate investigation of high-potential strains.
"""

import os
import pandas as pd
import numpy as np
import config

TARGET_KEYWORDS = [
    "fusaricidin",
    "polymyxin",
    "polymyxin_e1",
    "anabaenopeptin",
    "nostamide",
    "paenilipoheptin",
    "daunorubicin",
    "fosmidomycin",
    "teicoplanin",
    "clindamycin",
    "fosfomycin",
    "benzalkonium",
    "nitrogenase",
    "nifh",
    "nifd",
    "nifk",
    "polyamine",
    "putrescine",
    "fructoselysine",
    "frl",
    "cellobiose",
    "lichenan",
    "lic",
    "glucosidase",
]


def evaluate_strain_target_profile(df_matrix, target_keywords):
    """
    Scans gene column labels for user-defined functional keywords using
    vectorized string matching and computes a normalized target index.
    """
    # 1. Create a regex pattern (e.g., 'fusaricidin|polymyxin|nitrogenase')
    pattern = "|".join(target_keywords)

    # 2. Vectorized filter: returns a boolean mask for all columns at once
    # case=False ensures it matches regardless of capitalization
    mask = df_matrix.columns.str.contains(pattern, case=False, na=False)
    matched_genes = df_matrix.columns[mask]

    # 3. Guard clause: Handle case where no matches are found
    if matched_genes.empty:
        print("No target keyword signatures found in the current pangenome pool.")
        return pd.DataFrame(
            {"Target_Associated_Gene_Count": 0, "Target_Functional_Index": 0.0},
            index=df_matrix.index,
        )

    # 4. Extract sub-matrix and calculate profile
    df_target_matrix = df_matrix[matched_genes]
    strain_target_counts = df_target_matrix.sum(axis=1, numeric_only=True)

    # 5. Normalize score (0.0 to 1.0) based on the maximum found in current data
    max_detected = int(strain_target_counts.max())
    max_detected = max_detected if max_detected > 0 else 1
    target_index = strain_target_counts / max_detected

    # 6. Assemble and sort result
    df_profiles = pd.DataFrame(
        {
            "Target_Associated_Gene_Count": strain_target_counts,
            "Target_Functional_Index": np.round(target_index, 3),
        },
        index=df_matrix.index,
    )

    return df_profiles.sort_values(by="Target_Functional_Index", ascending=False)


def run_target_profiling(
    matrix_path=config.MATRIX_CSV,
    output_path=config.PATHOGENICITY_RISK_CSV,
    target_keywords=TARGET_KEYWORDS,
):
    """
    Main execution pathway for flexible functional profiling
    """
    if not os.path.exists(matrix_path):
        print("Error: Master matrix missing. Cannot compute profiles.")
        return None

    df_matrix = pd.read_csv(matrix_path, index_col=0)

    df_target_profiles = evaluate_strain_target_profile(df_matrix, target_keywords)

    df_target_profiles.to_csv(output_path)

    return df_target_profiles


if __name__ == "__main__":
    run_target_profiling()
