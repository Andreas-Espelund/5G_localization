import numpy as np


def haversine_distance(lat1, lon1, lat2, lon2) -> tuple[float, float, float]:
    """
    :param lat1: point a latitude
    :param lon1: point a longitude
    :param lat2: point b latitude
    :param lon2: point b longitude
    :return: distance in km, nmi and mi
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of Earth in kilometers (mean radius)
    r = 6371.0
    km = c * r
    nmi = km * 0.539956803  # nautical miles
    mi = km * 0.621371192  # miles
    return km
