import folium
import matplotlib.pyplot as plt
import pandas as pd


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


def make_boxplot(df: pd.DataFrame):
    """
    Creates a boxplot of the given dataframe.
    :param df:
    """
    data_values = [df[col] for col in df.columns]

    plt.figure(figsize=(20, 12))
    median_props = dict(color="black", linewidth="3")
    plot = plt.boxplot(
        data_values, patch_artist=True, labels=df.columns, medianprops=median_props
    )

    plt.title("MSE in Weighted Coverage strategy")
    plt.grid(axis="y")
    plt.xlabel("K value", fontsize=18)
    plt.ylabel("MSE (meters)", fontsize=18)

    # Show the plot
    plt.show()
