import math

import numpy as np


def gps2utm(lat, lon, lat0=49.015):  # also used by HA
    # from paper "Vision meets Robotics: The KITTI Dataset"

    r = 6378137.0
    s = np.cos(lat0 * np.pi / 180)

    x = s * r * np.pi * lon / 180
    y = s * r * np.log(np.tan(np.pi * (90 + lat) / 360))

    return x, y


# -------------------------------------------------------------
# WGS84 Constants
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # Eccentricity squared


def latlon_to_delta_xy(lat_ref, lon_ref, lat, lon):
    """
    Converts latitude/longitude differences into local XY displacements (meters).
    Uses an ellipsoidal Earth model (EPSG:4326 - WGS84) .

    Args:
        lat_ref (float): Reference latitude in degrees.
        lon_ref (float): Reference longitude in degrees.
        lat (float): Target latitude in degrees.
        lon (float): Target longitude in degrees.

    Returns:
        tuple: (dx, dy) displacement in meters (East, North)
    """
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    # Compute local radii of curvature
    R_N = WGS84_A / np.sqrt(
        1 - WGS84_E2 * np.sin(lat_ref_rad) ** 2
    )  # Transverse radius
    R_M = (WGS84_A * (1 - WGS84_E2)) / (
        1 - WGS84_E2 * np.sin(lat_ref_rad) ** 2
    ) ** 1.5  # Meridional radius

    # Compute displacements in meters
    dx = (
        R_N * np.cos(lat_ref_rad) * (np.radians(lon) - lon_ref_rad)
    )  # East displacement
    dy = R_M * (np.radians(lat) - lat_ref_rad)  # North displacement
    return dx, dy


# -------------------------------------------------------------

TILE = 256
EARTH_R = 6378137.0


def latlon_to_worldpx(lat, lon, z):
    lat = max(min(lat, 85.05112878), -85.05112878)
    siny = math.sin(math.radians(lat))
    x = (lon + 180.0) / 360.0
    y = 0.5 - 0.25 * math.log((1 + siny) / (1 - siny)) / math.pi
    scale = TILE * (2**z)
    return (x * scale, y * scale)


def pixel_in_image(lat0, lon0, latv, lonv, z, W, H, s):
    xc, yc = latlon_to_worldpx(lat0, lon0, z)
    xv, yv = latlon_to_worldpx(latv, lonv, z)
    dx, dy = xv - xc, yv - yc
    u = s * (W / 2.0 + dx)
    v = s * (H / 2.0 + dy)
    return u, v


def enu_from_latlon(lat0, lon0, latv, lonv):
    lat0r = math.radians(lat0)
    dlat = math.radians(latv - lat0)
    dlon = math.radians(lonv - lon0)
    E = EARTH_R * dlon * math.cos(lat0r)
    N = EARTH_R * dlat
    return E, N


def meters_per_pixel(lat0, z, s):
    return 156543.03392804097 * math.cos(math.radians(lat0)) / (2**z) / s


def H_uv_vehicle(lat0, lon0, latv, lonv, z, W, H, s, heading_deg=None):
    E, N = enu_from_latlon(lat0, lon0, latv, lonv)
    mpp = meters_per_pixel(lat0, z, s)
    k = 1.0 / mpp
    γ = math.radians(heading_deg)
    r11 = k * math.sin(γ)
    r12 = -k * math.cos(γ)
    r21 = k * math.cos(γ)
    r22 = k * math.sin(γ)
    tx = W * s / 2 + k * E
    ty = H * s / 2 - k * N
    return [[r11, r12, tx], [r21, r22, ty], [0, 0, 1]]
