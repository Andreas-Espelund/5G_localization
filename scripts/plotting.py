from typing import Union

import folium
import matplotlib.pyplot as plt
import numpy as np
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
    log_scale: bool = False,
    y_limit: tuple[float, float] = None,
    y_formatter=None,
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

    if log_scale:
        plt.yscale("log")

    if y_limit:
        plt.ylim(y_limit)

    def thousands_formatter(x, pos):
        return f"{x/1000:.0f}k"

    if y_formatter:
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(y_formatter))

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


def make_boxplot_double(
    df: pd.DataFrame,
    control_df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    legend_labels: list[str],
    color: Union[str, dict] = "forestgreen",
    baseline: float = None,
    annotate: bool = False,
):
    """
    Creates a boxplot of the given dataframes.

    :param legend_labels:
    :param control_df: DataFrame containing control data
    :param baseline: Optional baseline value to plot
    :param df: DataFrame containing experimental data
    :param title: Plot title
    :param x_label: X-axis label
    :param y_label: Y-axis label
    :param color: Color of the boxes (str or dict mapping column names to colors)
    :param annotate: Whether to annotate mean values
    """
    # Prepare data by pairing control and regular columns
    data_values = []
    labels = []
    means = []
    positions = []

    # Calculate positions for paired boxes
    for i in range(len(df.columns)):
        base_pos = i * 3  # Leave more space between pairs
        positions.extend([base_pos, base_pos + 1])  # Positions for control and regular

        col_name = df.columns[i]

        # Add control column
        data_values.append(control_df[col_name])
        labels.append(col_name)  # Use actual column name instead of number
        means.append(control_df[col_name].mean())

        # Add regular column
        data_values.append(df[col_name])
        labels.append(col_name)  # Use actual column name for the pair
        means.append(df[col_name].mean())

    num_boxes = len(data_values)

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values,
        positions=positions,  # Use custom positions
        patch_artist=True,
        medianprops=median_props,
    )

    # Handle color and hatch assignment
    legend_elements = []
    for i, patch in enumerate(plot["boxes"]):
        current_col = labels[i]
        current_color = color[current_col] if isinstance(color, dict) else color

        if i % 2 == 0:  # Control boxes
            patch.set_facecolor(current_color)
            patch.set_hatch("o")
            legend_elements.append(patch)
        else:  # Regular boxes
            patch.set_facecolor(current_color)
            patch.set_hatch("\\")
            legend_elements.append(patch)

    # Add legend
    plt.legend(legend_elements, legend_labels, loc="upper right")

    if baseline:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )

    if annotate:
        text_offset = 0.22 if num_boxes < 4 else 0.4
        for i, mean in enumerate(means):
            plt.text(
                positions[i] + text_offset,
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

    # Set x-ticks at the center of each pair
    tick_positions = [pos + 0.5 for pos in positions[::2]]  # Center between each pair
    plt.xticks(tick_positions, df.columns)

    # Show the plot
    plt.show()


def make_barplot_double(
    df: pd.DataFrame,
    control_df: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
    legend_labels: list[str],
    color: Union[str, dict] = "forestgreen",
    baseline: float = None,
    annotate: bool = False,
    logscale: bool = False,
):
    """
    Creates a bar plot with paired bars for control and experimental data.

    :param df: DataFrame containing experimental data
    :param control_df: DataFrame containing control data
    :param title: Title of the plot
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param legend_labels: Labels for the legend [control, experimental]
    :param color: Color for experimental bars (control bars will be lightblue)
    :param baseline: Optional baseline to be drawn across the plot
    :param annotate: Whether to annotate bar values
    """
    if df.empty or control_df.empty:
        raise ValueError("One of the DataFrames is empty.")

    # Calculate means
    exp_means = df.mean()
    control_means = control_df.mean()

    plt.figure(figsize=(20, 12))

    # Set the positions for the bars
    num_pairs = len(df.columns)
    bar_width = 0.35
    positions = np.arange(num_pairs) * 3  # Space between pairs

    # Create control bars
    control_bars = plt.bar(
        positions,
        control_means.values,
        bar_width,
        color="lightblue",
        hatch="/",
        label=legend_labels[0],
    )

    # Create experimental bars
    if isinstance(color, dict):
        colors = [color.get(col, "forestgreen") for col in df.columns]
        exp_bars = plt.bar(
            positions + bar_width,
            exp_means.values,
            bar_width,
            color=colors,
            hatch="\\",
            label=legend_labels[1],
        )
    else:
        exp_bars = plt.bar(
            positions + bar_width,
            exp_means.values,
            bar_width,
            color=color,
            hatch="\\",
            label=legend_labels[1],
        )

    if annotate:

        def annotate_bars(bars, values):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                )

        annotate_bars(control_bars, control_means)
        annotate_bars(exp_bars, exp_means)

    if baseline is not None:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )

    # Formatting the plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(axis="y")

    if logscale:
        plt.yscale("log")

    # Set x-ticks at the center of each pair
    plt.xticks(positions + bar_width / 2, df.columns)

    def thousands_formatter(x, pos):
        return f"{x/1000:.0f}k"

    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(thousands_formatter))

    plt.legend(loc="upper right")
    plt.margins(x=0.1)
    plt.tight_layout()

    plt.show()


def make_boxplot_grouped(
    df: pd.DataFrame,
    value_column: str,
    group_column: str,
    pair_column: str,
    title: str,
    x_label: str,
    y_label: str,
    legend_labels: list[str],
    color: Union[str, dict] = "forestgreen",
    baseline: float = None,
    annotate: bool = False,
):
    """
    Creates a grouped boxplot from a melted DataFrame.

    :param df: Melted DataFrame containing the data
    :param value_column: Name of the column containing the values to plot
    :param group_column: Name of the column to group by (x-axis categories)
    :param pair_column: Name of the column that defines the pairs (True/False or A/B etc.)
    :param title: Plot title
    :param x_label: X-axis label
    :param y_label: Y-axis label
    :param legend_labels: Labels for the legend [pair_false_label, pair_true_label]
    :param color: Color of the second box in each pair
    :param baseline: Optional baseline value to plot
    :param annotate: Whether to annotate mean values
    """
    # Get unique groups for positioning
    groups = df[group_column].unique()

    # Prepare data by grouping
    data_values = []
    positions = []
    means = []

    # Calculate positions for paired boxes
    for i, group in enumerate(groups):
        base_pos = i * 3  # Leave more space between pairs
        group_data = df[df[group_column] == group]

        # Get unique pair values
        pair_values = sorted(group_data[pair_column].unique())
        if len(pair_values) != 2:
            raise ValueError(
                f"Each group must have exactly 2 pair values. Group {group} has {len(pair_values)}"
            )

        # First box in pair
        first_data = group_data[group_data[pair_column] == pair_values[0]][value_column]
        data_values.append(first_data)
        positions.extend([base_pos])
        means.append(first_data.mean())

        # Second box in pair
        second_data = group_data[group_data[pair_column] == pair_values[1]][
            value_column
        ]
        data_values.append(second_data)
        positions.extend([base_pos + 1])
        means.append(second_data.mean())

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values,
        positions=positions,
        patch_artist=True,
        medianprops=median_props,
    )

    # Handle color and hatch assignment
    legend_elements = []
    for i, patch in enumerate(plot["boxes"]):
        if i % 2 == 0:  # First box in pair
            patch.set_facecolor("lightblue")
            patch.set_hatch("/")
            if i == 0:  # Only add to legend once
                legend_elements.append(patch)
        else:  # Second box in pair
            patch.set_facecolor(
                color
                if isinstance(color, str)
                else color.get(str(i // 2), "forestgreen")
            )
            patch.set_hatch("\\")
            if i == 1:  # Only add to legend once
                legend_elements.append(patch)

    # Add legend
    plt.legend(legend_elements, legend_labels, loc="upper right")

    if baseline:
        plt.axhline(
            y=baseline,
            color="red",
            linestyle="-",
            linewidth=2,
            label="Baseline",
        )

    if annotate:
        for pos, mean in zip(positions, means):
            plt.text(
                pos,
                mean,
                f"\n{mean:.2f}",
                horizontalalignment="center",
                verticalalignment="top",
                color="black",
            )

    # Formatting the plot
    plt.title(title)
    plt.grid(axis="y")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.margins(x=0.1)

    # Set x-ticks at the center of each pair
    tick_positions = [pos + 0.5 for pos in positions[::2]]  # Center between each pair
    plt.xticks(tick_positions, groups)

    plt.show()
