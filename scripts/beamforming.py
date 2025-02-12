from typing import Optional, Tuple

import pandas as pd

from scripts.utils import RF_PARAM_5G


def get_best_beam(
    mat: pd.DataFrame, rf_param: RF_PARAM_5G
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    # Drop rows where rf_param is NaN
    mat = mat.dropna(subset=[rf_param.value])

    # Check if the DataFrame is empty after dropping NaNs
    if mat.empty:
        print("No valid data available after dropping NaN values.")
        return None, None

    # Get the best beams by grouping only by 'pci' and 'operator_id'
    idx = mat.groupby(["pci"])[rf_param.value].idxmax()

    # Use the indices to select the rows with the highest 'rsrq' for each group
    best_beams = mat.loc[idx]

    # Get the best pci
    best_index = best_beams[rf_param.value].idxmax()
    best = best_beams.loc[best_index]

    # Make a tuple of pci, op, and beam (nr_arfcn is removed)
    pci = int(best["pci"])
    beam = int(best["beam_index"])

    return pci, beam


def filter_best_beams(
    mat: pd.DataFrame, rf_param: RF_PARAM_5G, group_by: [str] = ["pci"]
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    # Drop rows where rf_param is NaN
    mat = mat.dropna(subset=[rf_param.value])

    # Check if the DataFrame is empty after dropping NaNs
    if mat.empty:
        print("No valid data available after dropping NaN values.")
        return None, None

    # Get the best beams by grouping only by 'pci' and 'operator_id'
    idx = mat.groupby(group_by)[rf_param.value].idxmax()

    # Use the indices to select the rows with the highest 'rf-value' for each group
    best_beams = mat.loc[idx]

    return best_beams


def get_best_pci_beam_pairs(
    mat: pd.DataFrame, rf_param: RF_PARAM_5G
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    # Drop rows where rf_param is NaN
    mat = mat.dropna(subset=[rf_param.value])

    # Check if the DataFrame is empty after dropping NaNs
    if mat.empty:
        print("No valid data available after dropping NaN values.")
        return None, None

    # Get the best beams by grouping only by 'pci' and 'operator_id'
    idx = mat.groupby(["pci"])[rf_param.value].idxmax()

    # Use the indices to select the rows with the highest 'rsrq' for each group
    return mat.loc[idx][["pci", "beam_index"]].values


def find_matches(df_rp: pd.DataFrame, beam_pci: tuple[int, int]):
    return df_rp[df_rp["best_beam"] == beam_pci].index
