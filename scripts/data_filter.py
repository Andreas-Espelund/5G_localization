from typing import List

import pandas as pd


def filter_dataframe(
    df: pd.DataFrame,
    operators: List[int] = None,
    campaigns: List[int] = None,
    beams: List[int] = None,
    freqs: List[int] = None,
    include_columns: List[
        str
    ] = None,  # New parameter to specify which columns to include
) -> pd.DataFrame:
    # Function to filter an inner DataFrame based on the given criteria
    def filter_inner_df(inner_df):
        if operators is not None:
            inner_df = inner_df[inner_df["operator_id"].isin(operators)]

        if beams is not None:
            inner_df = inner_df[inner_df["beam_index"].isin(beams)]

        if freqs is not None:
            inner_df = inner_df[inner_df["nr_arfcn"].isin(freqs)]

        if include_columns is not None:
            inner_df = inner_df[include_columns]

        return inner_df

    if campaigns is not None:
        df = df[df["campaign_id"].isin(campaigns)]

    # Apply the filter function to each DataFrame in the 'measurement_matrix' column
    df["measurements_matrix"] = df["measurements_matrix"].apply(
        lambda x: filter_inner_df(x) if isinstance(x, pd.DataFrame) else x
    )
    # Remove rows where 'measurements_matrix' is empty
    df = df[
        df["measurements_matrix"].apply(
            lambda x: not (isinstance(x, pd.DataFrame) and x.empty)
        )
    ]

    return df
