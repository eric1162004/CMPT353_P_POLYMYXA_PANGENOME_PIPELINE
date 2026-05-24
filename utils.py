"""
CMPT353_P_POLYMYXA_PANGENOME_PIPELINE - Reusable Utilities
Shared helper functions across data engineering and analysis modules.
"""

import re
import pandas as pd


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
