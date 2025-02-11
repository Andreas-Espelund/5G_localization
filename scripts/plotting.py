from typing import Union

import folium
import matplotlib.pyplot as plt
import pandas as pd

from scripts.utils import cluster_color_mapping

plt.rcParams.update(
    {
        "font.size": 32,  # Default text size
        "axes.titlesize": 32,  # Title size
        "axes.labelsize": 32,  # X and Y label size
        "xtick.labelsize": 32,  # X tick label size
        "ytick.labelsize": 32,  # Y tick label size
        "legend.fontsize": 26,  # Legend text size
        "figure.titlesize": 32,  # Figure title size
    }
)

num_campaigns = 80
cmap = plt.get_cmap("viridis")
colors = [cmap(i / num_campaigns) for i in range(num_campaigns)]
campaign_colors = {i + 1: colors[i] for i in range(num_campaigns)}


color_maps = {"campaign_id": campaign_colors}


def geo_plot_points(df: pd.DataFrame):
    """
    Plots given locations to a map (OpenStreetMap) that is viewable in broswer.
    Generates a file called 'map.html' in the current working directory.
    :param df:
    """
    # Create a map centered around the mean location
    m = folium.Map(location=[df["lat"].mean(), df["lng"].mean()], zoom_start=12)

    # Add CircleMarkers to the map
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=5,  # Size of the marker
            color="blue",  # Border color of the marker
            fill=True,
            fill_color="blue",  # Fill color of the marker
            fill_opacity=0.6,
        ).add_to(m)

    # Save the map as an HTML file and open it in the browser
    m.save("map.html")


def make_boxplot(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    color: Union[str, dict] = "forestgreen",
    baseline: float = None,
    annotate: bool = False,
):
    """
    Creates a boxplot of the given dataframe.

    :param baseline:
    :param df:
    :param title:
    :param x_label:
    :param y_label:
    :param color: Color of the boxes
    """
    data_values = [df[col] for col in df.columns]
    means = [df[col].mean() for col in df.columns]
    num_boxes = len(df.columns)

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values,
        patch_artist=True,
        tick_labels=df.columns,
        medianprops=median_props,
    )

    # Handle color assignment
    if isinstance(color, dict):
        for patch, col_name in zip(plot["boxes"], df.columns):
            patch.set_facecolor(color.get(col_name, "forestgreen"))
    else:
        for patch in plot["boxes"]:
            patch.set_facecolor(color)

    if baseline:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )
        plt.legend(loc="upper right")

    if annotate:
        text_offset = 0.22 if num_boxes < 4 else 0.4

        for i, mean in enumerate(means, 1):
            plt.text(
                i + text_offset,
                mean,
                f"\n{mean:.2f}",
                horizontalalignment="right",
                verticalalignment="center",
                color="black",
            )

    # formatting the plot
    plt.title(title)
    plt.grid(axis="y")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.margins(x=0.1)

    # Show the plot
    plt.show()


def make_lat_lng_scatterplot_clustering(
    df: pd.DataFrame, col: str, col_label: str, title: str
):
    plt.figure(figsize=(18, 14))

    for name, group in df.groupby(col):
        plt.scatter(
            group["lat"],
            group["lng"],
            c=cluster_color_mapping[name],
            label=f"{col_label} {name}",
            alpha=0.8,
        )
    offset = 0.00025
    # plot the points that have been placed in the wrong cluster
    for cluster, group in df.groupby("cluster"):
        misplaced = group[group["prediction"] != cluster]
        for _, row in misplaced.iterrows():
            plt.scatter(row["lat"], row["lng"], c="black", alpha=1, marker="d", s=250)

            # top triangle show the predicted cluster, shown with its color
            plt.scatter(
                row["lat"],
                row["lng"] + offset,
                c=cluster_color_mapping[row["prediction"]],
                alpha=1,
                marker="^",
                s=100,
            )
            # bottom triangle show the correct cluster, shown with its color
            plt.scatter(
                row["lat"],
                row["lng"] - offset,
                c=cluster_color_mapping[cluster],
                alpha=1,
                marker="v",
                s=100,
            )

    ncol = 1
    if df[col].nunique() > 9:
        ncol = 2
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(title)
    plt.legend(title=col_label, loc="lower left", ncol=ncol)
    plt.show()


def make_barplot(
    df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    color: Union[str, dict] = "skyblue",
    baseline: float = None,
    annotate: bool = False,
):
    """
    Creates a bar plot of the given dataframe.

    :param df: DataFrame containing the data to be plotted
    :param title: Title of the plot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param color: Either a single color string for all bars or a dictionary mapping column names to colors
    :param baseline: Optional baseline to be drawn across the plot
    :param annotate: Whether to annotate bar values
    """
    # Ensure the DataFrame is not empty
    if df.empty:
        raise ValueError("The DataFrame is empty.")

    # Calculate the mean of each column to plot
    means = df.mean()

    plt.figure(figsize=(20, 12))

    # Handle color assignment
    if isinstance(color, dict):
        # Create a color list matching the order of means.index
        colors = [color.get(col, "skyblue") for col in means.index]
        bars = plt.bar(
            range(len(means)), means.values, color=colors, tick_label=means.index
        )
    else:
        # Use the same color for all bars if a single color is provided
        bars = plt.bar(
            range(len(means)), means.values, color=color, tick_label=means.index
        )

    if annotate:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
            )

    if baseline is not None:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )
        plt.legend(loc="upper right")

    # Formatting the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y")

    plt.margins(y=0.1)
    plt.tight_layout()

    # Show the plot
    plt.show()


def make_lat_lng_scatterplot(df: pd.DataFrame, col: str, col_label: str, title: str):
    plt.figure(figsize=(18, 14))

    # Use a discrete colormap with enough distinct colors
    num_campaigns = df[col].nunique()
    cmap = plt.get_cmap("tab20", num_campaigns)  # 'tab20' provides 20 distinct colors

    # Map each unique value to a color
    unique_values = df[col].unique()
    color_map = {val: cmap(i) for i, val in enumerate(unique_values)}

    for name, group in df.groupby(col):
        print(f"color {color_maps[col][name]}")
        plt.scatter(
            group["lat"],
            group["lng"],
            color=color_map[name],
            label=f"{col_label} {name}",
            alpha=0.8,
        )

    ncol = 1
    if df[col].nunique() > 9:
        ncol = 2
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.title(title)
    plt.legend(title=col_label, loc="lower left", ncol=ncol)
    plt.show()
