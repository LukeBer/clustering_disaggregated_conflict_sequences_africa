from math import asin, cos, radians, sin, sqrt

import numpy as np


def lat_lon_to_prio_gid(lat, lon):
    """
    Convert lat, lon to PRIO grid cell id based on 0.5 deg grid.
    PRIO grid starts at lat = -90, lon = -180.
    """
    # Calculate row and col index
    row = np.floor((lat + 90) * 2).astype(int)
    col = np.floor((lon + 180) * 2).astype(int)

    # Number of columns in the PRIO grid (360 / 0.5 = 720)
    n_cols = 720

    gid = row * n_cols + col
    return gid


def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r
