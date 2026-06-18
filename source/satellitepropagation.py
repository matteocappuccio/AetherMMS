"""GNSS almanac parsing, orbit propagation and local look-angle computation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

MU = 3.986005e14
OMEGA_E = 7.2921151467e-5
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


@dataclass
class Satellite:
    constellation: str
    prn: str
    sqrtA: float | None = None
    aSqRoot: float | None = None
    ecc: float = 0.0
    i0: float | None = None
    deltai: float = 0.0
    omega0: float = 0.0
    omegadot: float = 0.0
    w: float = 0.0
    m0: float = 0.0
    toa: float = 0.0
    wna: int = 0
    health: int = 0


def deg2rad(v: float) -> float:
    return v * math.pi / 180.0


def wrap_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def gps_week_and_tow(dt: datetime) -> tuple[int, float]:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = dt.astimezone(timezone.utc) - GPS_EPOCH
    sec = delta.total_seconds()
    week = int(sec // 604800)
    tow = float(sec - week * 604800)
    return week, tow


def _norm_key(k: str) -> str:
    """Normalize almanac field names such as 'SQRT(A)  (m 1/2)' and 'sqrtA'."""
    return re.sub(r"[^a-z0-9]+", "", k.lower())


def parse_yuma_almanac(path: Path) -> list[Satellite]:
    text = path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    sats: list[Satellite] = []
    for block in blocks:
        vals: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            raw = v.strip().split()
            if not raw:
                continue
            # Store both literal and normalized keys to support NAVCEN YUMA variants.
            vals[k.strip().lower()] = raw[0]
            vals[_norm_key(k)] = raw[0]
        if not vals:
            continue
        prn_raw = vals.get("id") or vals.get("prn") or vals.get("satellite id") or vals.get("satelliteid")
        if not prn_raw:
            continue
        try:
            prn_num = int(float(prn_raw))
        except ValueError:
            continue

        def f(*names: str, default: float | None = 0.0) -> float | None:
            for n in names:
                for key in (n.lower(), _norm_key(n)):
                    if key in vals:
                        try:
                            return float(vals[key].replace("D", "E"))
                        except ValueError:
                            pass
            return default

        sqrtA = f("sqrt(a)", "sqrt(a) (m 1/2)", "sqrta", "sqrt a", default=None)
        # A missing or zero sqrtA means the block was not parsed correctly; skip it rather than crashing.
        if sqrtA is None or not math.isfinite(float(sqrtA)) or float(sqrtA) <= 0.0:
            continue

        sats.append(Satellite(
            constellation="GPS", prn=f"G{prn_num:02d}",
            sqrtA=float(sqrtA),
            ecc=float(f("eccentricity", default=0.0) or 0.0),
            i0=float(f("orbital inclination(rad)", "orbital inclination", "inclination", default=deg2rad(55.0)) or deg2rad(55.0)),
            omega0=float(f("right ascen at week(rad)", "right ascension", "omega0", default=0.0) or 0.0),
            omegadot=float(f("rate of right ascen(r/s)", "rate of right ascen", "omegadot", default=0.0) or 0.0),
            w=float(f("argument of perigee(rad)", "argument of perigee", default=0.0) or 0.0),
            m0=float(f("mean anom(rad)", "mean anomaly", "m0", default=0.0) or 0.0),
            toa=float(f("time of applicability(s)", "toa(s)", "toa", default=0.0) or 0.0),
            health=int(float(f("health", default=0.0) or 0.0)),
        ))
    if not sats:
        raise ValueError(f"No valid GPS YUMA satellites parsed from {path}")
    return sats


def _txt(node, name: str) -> str | None:
    for el in node.iter():
        if el.tag.split("}")[-1] == name:
            return (el.text or "").strip()
    return None


def parse_galileo_xml(path: Path) -> list[Satellite]:
    root = ET.parse(path).getroot()
    sats: list[Satellite] = []
    for node in root.iter():
        if node.tag.split("}")[-1] != "svAlmanac":
            continue
        svid = _txt(node, "SVID")
        if not svid:
            continue
        def f(name: str, default: float = 0.0) -> float:
            t = _txt(node, name)
            try: return float(t) if t is not None else default
            except ValueError: return default
        n = int(svid)
        sats.append(Satellite(
            constellation="Galileo", prn=f"E{n:02d}", aSqRoot=f("aSqRoot"),
            ecc=f("ecc"), deltai=f("deltai") * math.pi, omega0=f("omega0") * math.pi,
            omegadot=f("omegaDot") * math.pi, w=f("w") * math.pi, m0=f("m0") * math.pi,
            toa=f("t0a"), wna=int(f("wna", 0.0)), health=int(f("statusE1B", 0.0))
        ))
    if not sats:
        raise ValueError(f"No Galileo satellites parsed from {path}")
    return sats


def solve_kepler(M: float, e: float, max_iter: int = 12) -> float:
    E = M
    for _ in range(max_iter):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    return E


def satellite_ecef(sat: Satellite, dt: datetime) -> np.ndarray | None:
    sqrtA = sat.sqrtA
    if sqrtA is None and sat.aSqRoot is not None:
        # Same convention used in the HTML: Galileo GSC XML aSqRoot is a delta around sqrt(29600 km).
        sqrtA = math.sqrt(29600000.0) + sat.aSqRoot
    if sqrtA is None or not math.isfinite(sqrtA):
        return None
    i0 = sat.i0 if sat.i0 is not None and math.isfinite(sat.i0) else deg2rad(56.0) + sat.deltai
    a = sqrtA * sqrtA
    if not math.isfinite(a) or a <= 0.0:
        return None
    n0 = math.sqrt(MU / (a * a * a))
    _, tow = gps_week_and_tow(dt)
    tk = tow - sat.toa
    if tk > 302400: tk -= 604800
    if tk < -302400: tk += 604800
    M = wrap_pi(sat.m0 + n0 * tk)
    E = solve_kepler(M, sat.ecc)
    v = math.atan2(math.sqrt(max(0.0, 1 - sat.ecc * sat.ecc)) * math.sin(E), math.cos(E) - sat.ecc)
    u = v + sat.w
    r = a * (1 - sat.ecc * math.cos(E))
    x_orb = r * math.cos(u)
    y_orb = r * math.sin(u)
    Omega = sat.omega0 + (sat.omegadot - OMEGA_E) * tk - OMEGA_E * sat.toa
    cosO, sinO = math.cos(Omega), math.sin(Omega)
    cosi, sini = math.cos(i0), math.sin(i0)
    return np.array([x_orb * cosO - y_orb * cosi * sinO, x_orb * sinO + y_orb * cosi * cosO, y_orb * sini], dtype=float)


def llh_to_ecef(lon_deg: float, lat_deg: float, h_m: float) -> np.ndarray:
    lon = deg2rad(lon_deg); lat = deg2rad(lat_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    x = (N + h_m) * math.cos(lat) * math.cos(lon)
    y = (N + h_m) * math.cos(lat) * math.sin(lon)
    z = (N * (1 - WGS84_E2) + h_m) * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def ecef_to_enu_vector(dx: np.ndarray, lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = deg2rad(lon_deg); lat = deg2rad(lat_deg)
    slon, clon = math.sin(lon), math.cos(lon)
    slat, clat = math.sin(lat), math.cos(lat)
    t = np.array([[-slon, clon, 0.0], [-slat * clon, -slat * slon, clat], [clat * clon, clat * slon, slat]])
    return t @ dx


def look_angles_from_sat(sat_ecef: np.ndarray, lon: float, lat: float, h: float) -> tuple[float, float, float]:
    recv = llh_to_ecef(lon, lat, h)
    enu = ecef_to_enu_vector(sat_ecef - recv, lon, lat)
    e, n, u = enu.tolist()
    horiz = math.hypot(e, n)
    az = (math.degrees(math.atan2(e, n)) + 360.0) % 360.0
    el = math.degrees(math.atan2(u, horiz))
    rng = float(np.linalg.norm(sat_ecef - recv))
    return az, el, rng


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


def load_satellites(example_dir: Path, inputs: dict, constellations: dict) -> list[Satellite]:
    sats: list[Satellite] = []
    if constellations.get("gps", True) and inputs.get("gps_yuma"):
        p = _resolve_path(example_dir, inputs["gps_yuma"])
        if p and p.exists():
            sats.extend(parse_yuma_almanac(p))
    if constellations.get("galileo", True) and inputs.get("galileo_xml"):
        p = _resolve_path(example_dir, inputs["galileo_xml"])
        if p and p.exists():
            sats.extend(parse_galileo_xml(p))
    if not sats:
        raise FileNotFoundError("No GNSS almanacs loaded. Check gps_yuma/galileo_xml paths and constellation flags.")
    return sats
