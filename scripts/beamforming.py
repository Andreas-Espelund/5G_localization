import numpy as np
import pandas as pd

from scripts.utils import RF_PARAM_5G


def get_best_beam(mat: pd.DataFrame, rf_param: RF_PARAM_5G):
    # Drop rows where rf_param is NaN
    mat = mat.dropna(subset=[rf_param.value])

    # Check if the DataFrame is empty after dropping NaNs
    if mat.empty:
        print("No valid data available after dropping NaN values.")
        return []

    # Get the best beams by grouping only by 'pci' and 'operator_id'
    idx = mat.groupby(["pci", "operator_id", "nr_arfcn"])[rf_param.value].idxmax()

    # Use the indices to select the rows with the highest 'rsrq' for each group
    best_beams = mat.loc[idx]

    # Get the best pci
    best_index = best_beams[rf_param.value].idxmax()
    best = best_beams.loc[best_index]

    return best["pci"], best["beam_index"], best["nr_arfcn"], best["operator_id"]


def filter_best_beams(
    mat: pd.DataFrame,
    rf_param: RF_PARAM_5G,
    group_by: [str] = ["pci"],
    threshold: float = None,
    use_sidelobes: bool = False,
) -> pd.DataFrame:
    # Drop rows where rf_param is NaN
    mat = mat.dropna(subset=[rf_param.value])

    # Check if the DataFrame is empty after dropping NaNs
    if mat.empty:
        print("No valid data available after dropping NaN values.")
        return None

    # Get the best beams by grouping by the specified columns
    idx = mat.groupby(group_by)[rf_param.value].idxmax()
    beams = mat.loc[idx]

    if threshold:
        max_val = mat[rf_param.value].max()
        beams = beams[beams[rf_param.value] >= max_val * threshold]

    if not use_sidelobes:
        return beams

    # get the sidelobes for the best beams
    result_beams = pd.DataFrame()
    for _, beam in beams.iterrows():
        sidelobes = get_sidelobe_rows(mat, beam)
        result_beams = pd.concat([result_beams, sidelobes], axis=0)

    return result_beams


def filter_best_beam(
    matrix: pd.DataFrame, rf_param: RF_PARAM_5G, use_sidelobes: bool = False
):
    matrix = matrix.dropna(subset=[rf_param.value])
    if matrix.empty:
        print("No valid data available after dropping NaN values.")
        return matrix

    # get the best beam
    idx = matrix[rf_param.value].idxmax()
    best_beam = matrix.loc[idx]

    if not use_sidelobes:
        return pd.DataFrame([best_beam])

    filtered_matrix = get_sidelobe_rows(matrix, best_beam)

    return filtered_matrix


def find_matching_rps(df_rp: pd.DataFrame, best_beam: np.array):
    # Convert comparison to check if arrays are equal element-wise
    mask = df_rp["best_beam"].apply(lambda x: x == best_beam)
    return df_rp[mask].index


def get_beam_sidelobes(beam_index: int) -> list[int]:
    prev_index = beam_index - 1 if beam_index > 0 else 7
    next_index = beam_index + 1 if beam_index < 7 else 0
    return [prev_index, beam_index, next_index]


def get_beam_sidelobe_pcis(beam: tuple) -> list[tuple]:
    pci, beam_index, nr_arfcn, operator_id = beam
    sidelobes = get_beam_sidelobes(beam_index)
    return [(pci, index, nr_arfcn, operator_id) for index in sidelobes]


def get_best_beam_diff(matrix: pd.DataFrame, rf_param: RF_PARAM_5G) -> np.float64:
    idx = matrix.groupby(["pci"])[rf_param.value].idxmax()
    beams = matrix.loc[idx]
    unique_sorted = beams[rf_param.value].sort_values(ascending=False)
    return unique_sorted.iloc[0] - unique_sorted.iloc[1]


def get_sidelobe_rows(matrix: pd.DataFrame, beam: pd.Series) -> pd.DataFrame:
    sidelobe_indices = get_beam_sidelobes(beam["beam_index"])

    return matrix[
        (matrix["pci"] == beam["pci"])
        & (matrix["operator_id"] == beam["operator_id"])
        & (matrix["nr_arfcn"] == beam["nr_arfcn"])
        & (matrix["beam_index"].isin(sidelobe_indices))
    ]


def get_all_beams_for_pci(matrix: pd.DataFrame, beam: pd.Series) -> pd.DataFrame:
    return matrix[
        (matrix["pci"] == beam["pci"])
        & (matrix["operator_id"] == beam["operator_id"])
        & (matrix["nr_arfcn"] == beam["nr_arfcn"])
    ]
