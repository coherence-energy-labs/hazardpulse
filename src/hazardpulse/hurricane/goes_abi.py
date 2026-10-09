"""GOES ABI band 13 (10.3 um clean longwave IR, 2 km) around a tropical cyclone, read straight from NOAA's public
S3 buckets by HTTP byte ranges: only the HDF5 chunks under the storm are transferred (~0.2 s per crop once the
file header is read), never the 23 MB full-disk file.

Each storm-hour is kept as a storm-centred POLAR image -- 200 radii (1-400 km, 2 km steps) x 64 azimuths of
brightness temperature, bilinear from the native fixed grid, uint8 at 0.6 K (170-322.4 K; 255 = missing) -- about 13 KB, enough for
the inner core's eye, eyewall and ring, and the representation every feature reads.

Fixed-grid geometry: GOES-R Product User Guide vol. 3, section 4.2.8 (the PUG's worked example is a test).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from typing import Iterable

import numpy as np

R_EQ = 6378137.0
R_POL = 6356752.31414
H_SAT = 35786023.0 + R_EQ
ECC = 0.0818191910435
N_R, DR_KM = 200, 2.0
N_AZ = 64
RADII_KM = (np.arange(N_R) + 0.5) * DR_KM            # 1, 3, ..., 399 km
AZIMUTHS = np.arange(N_AZ) * (2 * math.pi / N_AZ)    # 0 = north, clockwise
BT0, BT_STEP = 170.0, 0.6                            # uint8 v -> 170 + 0.6 v K (v <= 254, 322.4 K); 255 = missing
MISSING = 255
WEST_EAST_SPLIT = -106.2                             # midway between the GOES-West and GOES-East sub-points
BUCKET = "https://noaa-{sat}.s3.amazonaws.com"
PRODUCT = "ABI-L2-CMIPF"

# which satellite was operational in each slot (NOAA operational transitions)
EAST = ((dt.datetime(2017, 12, 18), "goes16"), (dt.datetime(2025, 4, 7), "goes19"))
WEST = ((dt.datetime(2019, 2, 12), "goes17"), (dt.datetime(2023, 1, 4), "goes18"))


def satellite_for(when: dt.datetime, lon: float) -> str:
    """The operational GOES that sees ``lon`` best at ``when`` (East for lon >= -106.2, West otherwise)."""
    slots = EAST if lon >= WEST_EAST_SPLIT or lon > 0 else WEST
    sat = None
    for start, name in slots:
        if when >= start:
            sat = name
    if sat is None:
        raise ValueError(f"no operational GOES for {when:%Y-%m-%d} at lon {lon}")
    return sat


def latlon_to_xy(lat, lon, lon0: float):
    """Scan angles (x, y, radians) of geodetic lat/lon (degrees) for a satellite at ``lon0``; NaN where the point
    is not visible from the satellite."""
    phi, lam = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    phic = np.arctan((R_POL ** 2 / R_EQ ** 2) * np.tan(phi))
    rc = R_POL / np.sqrt(1 - ECC ** 2 * np.cos(phic) ** 2)
    sx = H_SAT - rc * np.cos(phic) * np.cos(lam - math.radians(lon0))
    sy = -rc * np.cos(phic) * np.sin(lam - math.radians(lon0))
    sz = rc * np.sin(phic)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    y = np.arctan(sz / sx)
    hidden = H_SAT * (H_SAT - sx) < sy ** 2 + (R_EQ ** 2 / R_POL ** 2) * sz ** 2
    return np.where(hidden, np.nan, x), np.where(hidden, np.nan, y)


def xy_to_latlon(x, y, lon0: float):
    """Geodetic lat/lon (degrees) of scan angles (radians); NaN off the Earth's disk."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + (R_EQ ** 2 / R_POL ** 2) * np.sin(y) ** 2)
    b = -2.0 * H_SAT * np.cos(x) * np.cos(y)
    c = H_SAT ** 2 - R_EQ ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(disc)) / (2 * a)
    sx, sy, sz = rs * np.cos(x) * np.cos(y), -rs * np.sin(x), rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan((R_EQ ** 2 / R_POL ** 2) * sz / np.sqrt((H_SAT - sx) ** 2 + sy ** 2)))
    lon = lon0 - np.degrees(np.arctan(sy / (H_SAT - sx)))
    off = disc < 0
    return np.where(off, np.nan, lat), np.where(off, np.nan, lon)


def destination(lat: float, lon: float, bearing_rad: np.ndarray, dist_km: np.ndarray):
    """Great-circle destination points (degrees) from (lat, lon) along bearings (0 = north) at distances."""
    d = np.asarray(dist_km) / 6371.0
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(bearing_rad))
    l2 = l1 + np.arctan2(np.sin(bearing_rad) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 180.0) % 360.0 - 180.0


def polar_points(lat: float, lon: float):
    """Lat/lon of every polar sample (N_R x N_AZ) around a centre."""
    b, r = np.meshgrid(AZIMUTHS, RADII_KM)
    return destination(lat, lon, b, r)


def encode(bt: np.ndarray) -> np.ndarray:
    bt = np.asarray(bt, float)
    ok = np.isfinite(bt)
    v = np.rint((np.where(ok, bt, BT0) - BT0) / BT_STEP)
    out = np.clip(v, 0, 254).astype(np.uint8)
    out[~ok] = MISSING
    return out


def decode(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    bt = BT0 + BT_STEP * v.astype(float)
    bt[v == MISSING] = np.nan
    return bt


_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")


def list_keys(sat: str, hour: dt.datetime, fetch_text) -> list[str]:
    """Band-13 full-disk keys in one hour, sorted by scan start (``fetch_text(url) -> str``)."""
    prefix = f"{PRODUCT}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/OR_{PRODUCT}-M6C13"
    xml = fetch_text(f"{BUCKET.format(sat=sat)}/?list-type=2&prefix={prefix}")
    keys = _KEY_RE.findall(xml)
    if not keys:   # Mode 3 (15-min) before 2019-04-02 names files M3
        xml = fetch_text(f"{BUCKET.format(sat=sat)}/?list-type=2&prefix={prefix.replace('-M6C13', '-M3C13')}")
        keys = _KEY_RE.findall(xml)
    return sorted(keys, key=scan_start)


def scan_start(key: str) -> dt.datetime:
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    if not m:
        raise ValueError(key)
    y, doy, hh, mm, ss = map(int, m.groups())
    return dt.datetime(y, 1, 1) + dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def first_scan_at_or_after(keys: Iterable[str], hour: dt.datetime) -> str | None:
    for k in keys:
        if scan_start(k) >= hour - dt.timedelta(seconds=30):
            return k
    return None


class FullDisk:
    """An open ABI full-disk band-13 file (HTTP byte ranges): its fixed grid, and polar samples around centres."""

    def __init__(self, h5file):
        f = h5file
        self.x = f["x"][:] * float(f["x"].attrs["scale_factor"]) + float(f["x"].attrs["add_offset"])
        self.y = f["y"][:] * float(f["y"].attrs["scale_factor"]) + float(f["y"].attrs["add_offset"])
        self.lon0 = float(np.ravel(f["goes_imager_projection"].attrs["longitude_of_projection_origin"])[0])
        cmi = f["CMI"]
        self.cmi = cmi
        self.scale = float(np.ravel(cmi.attrs["scale_factor"])[0])
        self.offset = float(np.ravel(cmi.attrs["add_offset"])[0])
        self.fill = int(np.ravel(cmi.attrs["_FillValue"])[0])
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])          # negative: y decreases with row

    def polar(self, lat: float, lon: float) -> np.ndarray:
        """uint8 polar image (N_R x N_AZ) around (lat, lon); MISSING outside the disk or on fill."""
        plat, plon = polar_points(lat, lon)
        px, py = latlon_to_xy(plat, plon, self.lon0)
        ok = np.isfinite(px) & np.isfinite(py)
        out = np.full(plat.shape, MISSING, np.uint8)
        if not ok.any():
            return out
        fx = (px - self.x[0]) / self.dx
        fy = (py - self.y[0]) / self.dy
        c0, c1 = int(max(np.floor(np.nanmin(fx[ok])) - 1, 0)), int(min(np.ceil(np.nanmax(fx[ok])) + 2, len(self.x)))
        r0, r1 = int(max(np.floor(np.nanmin(fy[ok])) - 1, 0)), int(min(np.ceil(np.nanmax(fy[ok])) + 2, len(self.y)))
        if c1 - c0 < 2 or r1 - r0 < 2:
            return out
        raw = self.cmi[r0:r1, c0:c1]
        bt = np.where(raw == self.fill, np.nan, raw * self.scale + self.offset)
        gx, gy = fx[ok] - c0, fy[ok] - r0
        ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
        inside = (ix >= 0) & (iy >= 0) & (ix + 1 < bt.shape[1]) & (iy + 1 < bt.shape[0])
        val = np.full(gx.shape, np.nan)
        wx, wy = gx[inside] - ix[inside], gy[inside] - iy[inside]
        a, b = bt[iy[inside], ix[inside]], bt[iy[inside], ix[inside] + 1]
        c, d = bt[iy[inside] + 1, ix[inside]], bt[iy[inside] + 1, ix[inside] + 1]
        val[inside] = (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy)
        enc = np.full(plat.shape, np.nan)
        enc[ok] = val
        return encode(enc)


EYE_SEARCH_KM = 25.0      # an eye is looked for within this distance of the analysed centre
EYE_CONTRAST_K = 15.0     # ... and counts only if this much warmer than the coldest azimuthal mean within 100 km


def find_eye(img: np.ndarray) -> tuple[float, float] | None:
    """(bearing rad, distance km) of a clear eye in a polar image -- the warmest sample within EYE_SEARCH_KM, if at
    least EYE_CONTRAST_K warmer than the coldest azimuthal mean within 100 km -- else None. The analysed centre
    is a few km off a pinhole eye (position error, cloud-top parallax), and inner-core features need the eye."""
    bt = decode(img)
    inner = RADII_KM <= EYE_SEARCH_KM
    block = bt[inner]
    if not np.isfinite(block).any():
        return None
    with np.errstate(all="ignore"):
        ring = np.nanmean(bt[(RADII_KM >= 1.0) & (RADII_KM <= 100.0)], axis=1)
    if not np.isfinite(ring).any():
        return None
    i, j = np.unravel_index(np.nanargmax(block), block.shape)
    if block[i, j] - np.nanmin(ring) < EYE_CONTRAST_K:
        return None
    return float(AZIMUTHS[j]), float(RADII_KM[inner][i])


ENCLOSE_K = 235.0         # an eye is ringed by cloud at least this cold ...
ENCLOSE_KM = 50.0         # ... within this distance of it ...
ENCLOSE_FRAC = 0.75       # ... in at least this fraction of directions (an exposed low-level centre is not)


def enclosed(img: np.ndarray) -> float:
    """The fraction of directions around a polar image's centre that reach cloud colder than ENCLOSE_K within
    ENCLOSE_KM (from 4 km out)."""
    bt = decode(img)
    band = bt[(RADII_KM >= 4.0) & (RADII_KM <= ENCLOSE_KM)]
    with np.errstate(all="ignore"):
        coldest = np.nanmin(np.where(np.isfinite(band), band, np.inf), axis=0)
    return float(np.mean(coldest <= ENCLOSE_K))


def polar_recentred(disk: "FullDisk", lat: float, lon: float) -> tuple[np.ndarray, float, float, bool]:
    """(polar image, centre lat, centre lon, eye found): the polar image around the analysed centre, re-sampled
    around a clear eye when ``find_eye`` finds one AND cold cloud encloses it (``enclosed`` >= ENCLOSE_FRAC). A warm
    spot beside one-sided convection -- an exposed centre -- keeps the analysed centre. Deterministic: the same in
    training and live."""
    img = disk.polar(lat, lon)
    eye = find_eye(img)
    if eye is None:
        return img, lat, lon, False
    elat, elon = destination(lat, lon, np.array(eye[0]), np.array(eye[1]))
    elat, elon = float(elat), float(elon)
    img_eye = disk.polar(elat, elon)
    if enclosed(img_eye) < ENCLOSE_FRAC:
        return img, lat, lon, False
    return img_eye, elat, elon, True


def open_full_disk(sat: str, key: str):
    """(file handle, h5py.File) for a full-disk key, read by HTTP byte ranges. Close both when done."""
    import fsspec
    import h5py
    fh = fsspec.filesystem("http").open(f"{BUCKET.format(sat=sat)}/{key}", "rb", block_size=2 ** 20,
                                        cache_type="blockcache")
    return fh, h5py.File(fh, "r")
