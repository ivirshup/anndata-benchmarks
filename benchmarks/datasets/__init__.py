"""This module should hold the code to read in datasets I'll use for benchmarking."""

from .immune_cell_atlas import ICA_BoneMarrow_full, ICA_BoneMarrow_Donor1

DATASETS = [ICA_BoneMarrow_full, ICA_BoneMarrow_Donor1]


def list_available(datasets=DATASETS):
    """List datasets currently setup."""
    return list(filter(lambda x: x.is_available(), datasets))