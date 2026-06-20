"""AetherMMS city-building module.

This module mirrors the data-ingestion / urban reconstruction part of the
original AetherMMS HTML/JavaScript implementation:

- DTM is mandatory and is sampled along the route and obstruction features.
- Buildings and trees are filtered to the MMS LiDAR buffer.
- OSM tunnel / covered road / overpass querying is optional and controlled by
  config/config.txt.
- Geometry is pre-indexed in a local metric CRS so the GNSS ray-tracing module
  can reproduce the original ray tests without doing an all-features loop for
  every satellite.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import bisect
import json
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as crs_transform
import requests
from pyproj import CRS, Transformer, Geod
from shapely.geometry import LineString, Point, shape, mapping
from shapely.ops import transform as shp_transform
from shapely.strtree import STRtree


WGS84 = CRS.from_epsg(4326)
GEOD = Geod(ellps="WGS84")
TURF_EARTH_RADIUS_M = 6371008.8
_TURF_ROUTE_CACHE: dict[int, tuple[list[tuple[float, float]], list[float], float]] = {}


@dataclass
class DemModel:
    path: Path
    crs: CRS
    transform: Any
    nodata: float | None
    dataset: Any
    array: Any
    inv_transform: Any
    xy_transformer: Transformer | None = None
    route_base_m: float = 0.0
    res_m: float = 5.0

    def sample(self, lon: float, lat: float) -> float:
        """Sample absolute DTM elevation at lon/lat. Returns NaN outside/nodata.

        Pixel selection matches the WebUI sampler exactly: the fractional
        edge-based pixel coordinate is rounded half-up (JS Math.round) and the
        point is rejected when it falls beyond width-1/height-1, instead of the
        usual GDAL floor() cell lookup. The two conventions differ by half a
        cell and would otherwise desynchronise building/tree bases.
        """
        try:
            if self.xy_transformer is not None:
                x, y = self.xy_transformer.transform(lon, lat)
            else:
                x, y = lon, lat
            c_f, r_f = self.inv_transform * (x, y)
            h, w = self.array.shape
            if not (math.isfinite(c_f) and math.isfinite(r_f)) or c_f < 0 or r_f < 0 or c_f > w - 1 or r_f > h - 1:
                return float("nan")
            c = min(w - 1, max(0, int(math.floor(c_f + 0.5))))
            r = min(h - 1, max(0, int(math.floor(r_f + 0.5))))
            val = float(self.array[r, c])
            if self.nodata is not None and val == float(self.nodata):
                return float("nan")
            return val if math.isfinite(val) else float("nan")
        except Exception:
            return float("nan")

    def sample_relative(self, lon: float, lat: float) -> float:
        """Equivalent of the HTML DTM sampleRelative(): elevation relative to route mean."""
        z = self.sample(lon, lat)
        return z - self.route_base_m if math.isfinite(z) else float("nan")

    def contains_lonlat(self, lon: float, lat: float) -> bool:
        return math.isfinite(self.sample(lon, lat))

    def close(self) -> None:
        try:
            self.dataset.close()
        except Exception:
            pass


@dataclass
class MetricScene:
    """Precomputed metric scene used by ray tracing.

    Coordinates are in a local ENU / tangent-plane CRS (azimuthal equidistant)
    centred on the route midpoint, so the North axis is true North and the ENU
    satellite azimuths are reused as planar bearings, faithfully to the Turf.js
    HTML prototype. The original HTML uses Turf bbox filtering and line/footprint
    intersections; this object provides the same logic efficiently in Python.
    """
    fwd: Transformer
    inv: Transformer
    route_m: LineString
    building_geoms: list[Any]
    building_lonlat_geoms: list[Any]
    building_props: list[dict[str, Any]]
    building_tree: STRtree | None
    tree_props: list[dict[str, Any]]
    tunnel_geoms: list[Any]
    tunnel_tree: STRtree | None


def load_dem(path: Path) -> DemModel:
    if not path.exists():
        raise FileNotFoundError(f"DTM GeoTIFF not found: {path}")
    ds = rasterio.open(path)
    if ds.crs is None:
        ds.close()
        raise ValueError("DTM has no readable CRS. Provide a georeferenced GeoTIFF.")
    arr = ds.read(1, masked=False)
    crs = CRS.from_user_input(ds.crs)
    xy_transformer = None if crs == WGS84 else Transformer.from_crs(WGS84, crs, always_xy=True)
    # Ground sampling distance of the DTM, in metres. The terrain ray-tracing
    # samples the profile at this resolution (Section 2.6 of the manual).
    res_m = abs(float(ds.transform.a))
    if crs.is_geographic:
        res_m *= 111320.0  # degrees -> metres (approximate, for lon/lat DTMs)
    if not math.isfinite(res_m) or res_m <= 0:
        res_m = 5.0
    return DemModel(path=path, crs=crs, transform=ds.transform, nodata=ds.nodata, dataset=ds, array=arr, inv_transform=(~ds.transform), xy_transformer=xy_transformer, res_m=res_m)


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def parse_kml_lines(path: Path) -> list[LineString]:
    if not path.exists():
        raise FileNotFoundError(f"KML file not found: {path}")
    root = ET.parse(path).getroot()
    lines: list[LineString] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) != "coordinates" or not elem.text:
            continue
        coords = []
        for token in elem.text.replace("\n", " ").split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        if len(coords) >= 2:
            lines.append(LineString(coords))
    if not lines:
        raise ValueError(f"No LineString coordinates found in KML: {path}")
    return lines


def parse_kml_points(path: Path | None) -> list[Point]:
    if not path or not path.exists():
        return []
    pts: list[Point] = []
    root = ET.parse(path).getroot()
    for elem in root.iter():
        if _strip_ns(elem.tag) != "coordinates" or not elem.text:
            continue
        tokens = elem.text.replace("\n", " ").split()
        if len(tokens) == 1:
            parts = tokens[0].split(",")
            if len(parts) >= 2:
                try:
                    pts.append(Point(float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return pts


def extract_longest_route(path: Path) -> LineString:
    return max(parse_kml_lines(path), key=lambda g: len(g.coords))


def load_geojson(path: Path | None, required: bool = False) -> dict[str, Any]:
    if path is None or str(path).lower() in {"none", "null", ""}:
        if required:
            raise FileNotFoundError("Required GeoJSON path is missing")
        return {"type": "FeatureCollection", "features": []}
    if not path.exists():
        if required:
            raise FileNotFoundError(f"GeoJSON not found: {path}")
        return {"type": "FeatureCollection", "features": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") == "Feature":
        data = {"type": "FeatureCollection", "features": [data]}
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Unsupported GeoJSON type in {path}: {data.get('type')}")
    return data


def local_metric_crs_for_geometry(geom: LineString) -> CRS:
    """Local ENU / tangent-plane CRS centred on the route midpoint.

    The engine derives satellite azimuth/elevation in the local East-North-Up
    frame (true North). The obstruction ray-tracing and the planimetric figures
    must live in that same frame, so we use an azimuthal-equidistant projection
    centred on the route midpoint instead of a UTM zone. Its North axis is true
    North (no meridian/grid convergence), therefore the ENU azimuth is reused as
    a planar bearing exactly as the Turf-based HTML prototype did, and distances
    are metric and locally exact. This removes the ~1.6 deg UTM grid rotation at
    Bologna that would otherwise mis-point every ray by up to a few tens of
    metres over the 1.2 km ray length.
    """
    lon, lat = geom.interpolate(0.5, normalized=True).coords[0]
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat:.10f} +lon_0={lon:.10f} "
        f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )


def transformers(route: LineString) -> tuple[Transformer, Transformer]:
    crs = local_metric_crs_for_geometry(route)
    return Transformer.from_crs(WGS84, crs, always_xy=True), Transformer.from_crs(crs, WGS84, always_xy=True)


def to_metric(geom, fwd: Transformer):
    return shp_transform(lambda x, y, z=None: fwd.transform(x, y), geom)


def to_wgs84(geom, inv: Transformer):
    return shp_transform(lambda x, y, z=None: inv.transform(x, y), geom)


def _lonlat2(coord) -> tuple[float, float]:
    return float(coord[0]), float(coord[1])


def _turf_distance_m(a, b) -> float:
    lon1, lat1 = map(math.radians, _lonlat2(a))
    lon2, lat2 = map(math.radians, _lonlat2(b))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * TURF_EARTH_RADIUS_M * math.atan2(math.sqrt(h), math.sqrt(max(0.0, 1.0 - h)))


def _turf_bearing_deg(a, b) -> float:
    lon1, lat1 = map(math.radians, _lonlat2(a))
    lon2, lat2 = map(math.radians, _lonlat2(b))
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _turf_destination(a, dist_m: float, bearing_deg: float) -> tuple[float, float]:
    lon1, lat1 = map(math.radians, _lonlat2(a))
    brng = math.radians(float(bearing_deg))
    delta = float(dist_m) / TURF_EARTH_RADIUS_M
    lat2 = math.asin(math.sin(lat1) * math.cos(delta) + math.cos(lat1) * math.sin(delta) * math.cos(brng))
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(delta) * math.cos(lat1),
        math.cos(delta) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lon2) + 540.0) % 360.0 - 180.0, math.degrees(lat2)


def turf_route_length_m(route: LineString) -> float:
    """Match Turf.length(route, {units:'kilometers'}) * 1000 used by the WebUI."""
    return _turf_route_data(route)[2]


def turf_route_position_at_distance(route: LineString, distance_m: float) -> tuple[float, float]:
    """Match Turf.along(route, distance/1000, {units:'kilometers'})."""
    coords, cumulative, total_m = _turf_route_data(route)
    target = max(0.0, float(distance_m))
    if target <= 0.0:
        return coords[0]
    if target >= total_m:
        return coords[-1]
    i = max(1, bisect.bisect_left(cumulative, target))
    return _turf_destination(coords[i - 1], target - cumulative[i - 1], _turf_bearing_deg(coords[i - 1], coords[i]))


def _turf_route_data(route: LineString) -> tuple[list[tuple[float, float]], list[float], float]:
    key = id(route)
    cached = _TURF_ROUTE_CACHE.get(key)
    if cached is not None:
        return cached
    coords = [_lonlat2(c) for c in route.coords]
    if not coords:
        raise ValueError("Route has no coordinates")
    cumulative = [0.0]
    for a, b in zip(coords, coords[1:]):
        cumulative.append(cumulative[-1] + _turf_distance_m(a, b))
    cached = (coords, cumulative, float(cumulative[-1]))
    if len(_TURF_ROUTE_CACHE) > 16:
        _TURF_ROUTE_CACHE.clear()
    _TURF_ROUTE_CACHE[key] = cached
    return cached


def route_length_m(route: LineString) -> float:
    return turf_route_length_m(route)


def metric_route_length_m(route: LineString) -> float:
    fwd, _ = transformers(route)
    return float(to_metric(route, fwd).length)


def buffer_route(route: LineString, buffer_m: float):
    fwd, inv = transformers(route)
    return to_wgs84(to_metric(route, fwd).buffer(buffer_m), inv)


def _feature_shape(feature: dict[str, Any]):
    return shape(feature["geometry"])


def _numeric_prop(props: dict[str, Any], names: Iterable[str], default: float = 0.0) -> float:
    for name in names:
        if name in props and props[name] not in (None, ""):
            try:
                v = float(str(props[name]).replace(",", "."))
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return default


def _js_number(value: Any) -> float:
    """JS Number() semantics for attribute guessing: null -> 0, non-numeric -> NaN."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else float("nan")
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float("nan")


def _js_coalesce_number(props: dict[str, Any], names: Iterable[str]) -> float:
    """Equivalent of Number(p.a ?? p.b ?? ...): first present non-null key wins."""
    for n in names:
        if n in props and props[n] is not None:
            return _js_number(props[n])
    return float("nan")


def _guess_building_height(props: dict[str, Any]) -> float:
    """Port of HTML guessHeight(): exact field list, quota_gronda-quota_piede
    difference, then any *height*/h_avg field, fallback 16 m."""
    keys = list((props or {}).keys())
    exact = ["height_gr", "height", "h", "quota_gron", "quota_pied", "quota_gronda", "gronda", "z_max", "zmax", "h_edif"]
    for name in exact:
        k = next((x for x in keys if str(x).lower() == name), None)
        if k is None:
            continue
        v = _js_number(props.get(k))
        if math.isfinite(v) and 0.0 < v < 200.0:
            # The HTML skips the absolute quota fields here: they are elevations,
            # not heights, and are only used through the qg-qp difference below.
            if name in ("quota_gron", "quota_pied", "quota_gronda"):
                continue
            return v
    qg = _js_coalesce_number(props, ["quota_gron", "QUOTA_GRON", "quota_gronda", "QUOTA_GRONDA"])
    qp = _js_coalesce_number(props, ["quota_pied", "QUOTA_PIED", "quota_piede", "QUOTA_PIEDE"])
    if math.isfinite(qg) and math.isfinite(qp) and qg > qp and qg - qp < 200.0:
        return qg - qp
    for k in keys:
        u = str(k).lower()
        v = _js_number(props.get(k))
        if ("height" in u or "h_avg" in u) and math.isfinite(v) and 0.0 < v < 200.0:
            return v
    return 16.0


def normalize_buildings(buildings: dict[str, Any], route_buffer) -> dict[str, Any]:
    out = []
    for f in buildings.get("features", []):
        if not f.get("geometry"):
            continue
        try:
            geom = _feature_shape(f)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or not geom.intersects(route_buffer):
                continue
            props = dict(f.get("properties") or {})
            h = _guess_building_height(props)
            props["_height"] = max(0.0, h)
            out.append({"type": "Feature", "geometry": mapping(geom), "properties": props})
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": out}


def _norm_name(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(value).lower())


def _parse_numeric_range_min(raw: Any, max_value: float = 500.0) -> float | None:
    """Python port of HTML parseNumericRangeMin()."""
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.floating)):
        v = float(raw)
        return v if math.isfinite(v) and 0.0 < v < max_value else None
    s = str(raw).strip().replace(",", ".")
    if not s or re.match(r"^(null|undefined|nan)$", s, flags=re.I):
        return None
    if ":" in s:
        s = ":".join(s.split(":")[1:]).strip()
    s = re.sub(r"^\s*(cl|classe)\s*\d+\s*", "", s, flags=re.I).strip()
    vals = [float(m.group(0)) for m in re.finditer(r"\d+(?:\.\d+)?", s)]
    vals = [v for v in vals if math.isfinite(v) and 0.0 < v < max_value]
    return min(vals) if vals else None


def _parse_numeric_range_mid(raw: Any, max_value: float = 500.0) -> float | None:
    """Python port of HTML parseNumericRangeMid()."""
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.floating)):
        v = float(raw)
        return v if math.isfinite(v) and 0.0 < v < max_value else None
    s = str(raw).strip().replace(",", ".")
    if not s or re.match(r"^(null|undefined|nan)$", s, flags=re.I):
        return None
    if ":" in s:
        s = ":".join(s.split(":")[1:]).strip()
    s = re.sub(r"^\s*(cl|classe)\s*\d+\s*", "", s, flags=re.I).strip()
    vals = [float(m.group(0)) for m in re.finditer(r"\d+(?:\.\d+)?", s)]
    vals = [v for v in vals if math.isfinite(v) and 0.0 < v < max_value]
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return (vals[0] + vals[1]) / 2.0


def _parse_tree_height_value(raw: Any) -> float | None:
    """Python port of HTML parseTreeHeightValue().

    Important: this is intentionally non-estimating. If no readable height is
    present, the tree height is 0, exactly like the HTML guessTreeHeight().
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.floating)):
        v = float(raw)
        return v if math.isfinite(v) and 0.0 < v < 80.0 else None
    s = str(raw).strip().replace(",", ".")
    if not s or re.match(r"^(null|undefined|nan)$", s, flags=re.I):
        return None
    if ":" in s:
        s = ":".join(s.split(":")[1:]).strip()
    s = re.sub(r"^\s*(cl|classe)\s*\d+\s*", "", s, flags=re.I).strip()
    vals = [float(m.group(0)) for m in re.finditer(r"\d+(?:\.\d+)?", s)]
    vals = [v for v in vals if math.isfinite(v) and 0.0 < v < 80.0]
    return min(vals) if vals else None


def _tree_height_field_value(props: dict[str, Any]) -> dict[str, Any]:
    keys = list((props or {}).keys())
    preferred = [
        "height", "HEIGHT", "ALTEZZA", "h", "H", "tree_height", "TREE_HEIGHT",
        "height_tot", "TOTAL_HEIGHT", "height_m", "ALTEZZA_M", "height_mt", "ALTEZZA_MT",
        "cl_h", "CL_H", "classe_h", "CLASSE_H", "classe_height", "CLASSE_ALTEZZA",
        "classe_ht", "CLASSE_HT", "classeheight", "CLASSEALTEZZA", "cl_height", "CL_ALTEZZA",
        "classe_di_height", "CLASSE_DI_ALTEZZA", "classe height",
    ]
    for name in preferred:
        k = next((x for x in keys if _norm_name(x) == _norm_name(name)), None)
        if not k:
            continue
        v = _parse_tree_height_value(props.get(k))
        if v is not None and math.isfinite(v):
            return {"value": v, "source": f"{k}: {props.get(k)}"}
    for k in keys:
        if not re.search(r"(height|tree_height|h_tree|hchioma|h_chioma|h$|^h$)", str(k), flags=re.I):
            continue
        v = _parse_tree_height_value(props.get(k))
        if v is not None and math.isfinite(v):
            return {"value": v, "source": f"{k}: {props.get(k)}"}
    return {"value": None, "source": "no readable height field"}


def _tree_trunk_diameter_field_value(props: dict[str, Any]) -> dict[str, Any]:
    keys = list((props or {}).keys())
    preferred = [
        "classe_circonferenza_diametro", "CLASSE_CIRCONFERENZA_DIAMETRO",
        "diametro", "DIAMETRO", "diametro_tronco", "DIAMETRO_TRONCO",
        "diameter", "DIAMETER", "dbh", "DBH", "diameter_cm", "DIAMETER_CM",
        "circonferenza_diametro", "CIRCONFERENZA_DIAMETRO",
    ]
    for name in preferred:
        k = next((x for x in keys if _norm_name(x) == _norm_name(name)), None)
        if not k:
            continue
        raw = props.get(k)
        txt = str(raw if raw is not None else "")
        v = None
        match = re.search(r"\(([^)]*)\)", txt)
        if match:
            v = _parse_numeric_range_mid(match.group(1), 300.0)
        if v is None or not math.isfinite(v):
            v = _parse_numeric_range_mid(raw, 300.0)
        if v is not None and math.isfinite(v):
            cm = v * 100.0 if v <= 3.0 else v
            return {"value": cm, "source": f"{k}: {raw}"}
    return {"value": None, "source": "no readable diameter field"}


def _guess_crown_diameter(props: dict[str, Any]) -> dict[str, Any]:
    keys = list((props or {}).keys())
    candidates = [
        "diameter_crown", "crown_diameter", "diametro_chioma", "DIAMETRO_CHIOMA",
        "chioma", "CHIOMA", "crown", "diam_chioma", "DIAM_CHIOMA", "ampiezza_chioma",
        "raggio_chioma", "RAGGIO_CHIOMA",
    ]
    for name in candidates:
        k = next((x for x in keys if _norm_name(x) == _norm_name(name)), None)
        if not k:
            continue
        v = _parse_numeric_range_min(props.get(k), 80.0)
        if v is not None and math.isfinite(v) and 0.0 < v < 80.0:
            return {
                "diameter": v * 2.0 if re.search(r"(raggio|radius)", k, flags=re.I) else v,
                "source": f"{k}: {props.get(k)}",
                "mode": "crown-from-field",
            }
    trunk = _tree_trunk_diameter_field_value(props)
    trunk_value = trunk.get("value")
    if isinstance(trunk_value, (int, float, np.floating)) and math.isfinite(float(trunk_value)):
        d_cm = float(trunk_value)
        crown_m = max(2.5, min(10.0, d_cm / 7.0))
        return {
            "diameter": crown_m,
            "source": f"crown modeled from {trunk.get('source')}",
            "mode": "crown-estimated-from-trunk-diameter",
            "trunkDiameterCm": d_cm,
            "trunkSource": trunk.get("source"),
        }
    return {"diameter": 0.0, "source": "no readable diameter", "mode": "no-diameter"}


def _circle_lonlat(lon: float, lat: float, radius_m: float, steps: int = 24):
    """Circle equivalent to turf.circle(..., units='meters').

    turf.circle builds the ring with turf.destination (spherical), so we reuse the
    same spherical destination used elsewhere in this module instead of pyproj's
    Karney geodesic. This is both faithful to the WebUI and far cheaper: the
    geodesic forward solver dominated tree pre-processing for large datasets.
    """
    from shapely.geometry import Polygon
    coords = [_turf_destination((lon, lat), float(radius_m), 360.0 * i / steps) for i in range(steps)]
    coords.append(coords[0])
    return Polygon(coords)


def _tree_feature_to_crown(feature: dict[str, Any], idx: int) -> dict[str, Any] | None:
    props_in = dict(feature.get("properties") or {})
    h_info = _tree_height_field_value(props_in)
    h_val = h_info.get("value")
    h = float(h_val) if isinstance(h_val, (int, float, np.floating)) and math.isfinite(float(h_val)) else 0.0
    crown = _guess_crown_diameter(props_in)
    d = float(crown.get("diameter") or 0.0)
    radius_m = max(0.4, d / 2.0)
    trunk_info = _tree_trunk_diameter_field_value(props_in)
    trunk_diam_raw = crown.get("trunkDiameterCm") or trunk_info.get("value") or 0.0
    trunk_diameter_cm = float(trunk_diam_raw) if isinstance(trunk_diam_raw, (int, float, np.floating)) and math.isfinite(float(trunk_diam_raw)) else 0.0
    trunk_radius_m = max(0.08, min(0.45, trunk_diameter_cm / 200.0 if trunk_diameter_cm > 0.0 else radius_m * 0.08))
    crown_base_rel = max(1.8, min(h * 0.60, h * 0.38))
    crown_height = max(0.5, h - crown_base_rel)

    try:
        geom = _feature_shape(feature)
        if geom.is_empty:
            return None
        gt = geom.geom_type
        center = None
        crown_geom = None
        if gt == "Point":
            center = (geom.x, geom.y)
            crown_geom = _circle_lonlat(center[0], center[1], radius_m, steps=24)
        elif gt == "MultiPoint":
            first = list(geom.geoms)[0]
            center = (first.x, first.y)
            crown_geom = _circle_lonlat(center[0], center[1], radius_m, steps=24)
        elif gt in {"LineString", "MultiLineString"}:
            center = _turf_centroid_lonlat(feature["geometry"])
            crown_geom = _circle_lonlat(center[0], center[1], radius_m, steps=24) if center else None
        elif gt in {"Polygon", "MultiPolygon"}:
            center = _turf_centroid_lonlat(feature["geometry"])
            crown_geom = geom
        else:
            return None
        if crown_geom is None or center is None or crown_geom.is_empty:
            return None
        if not crown_geom.is_valid:
            crown_geom = crown_geom.buffer(0)
        props = dict(props_in)
        props.update({
            "_tree_center_lon": float(center[0]),
            "_tree_center_lat": float(center[1]),
            "_tree_height": h,
            "_crown_diameter": d,
            "_crown_radius_m": radius_m,
            "_crown_base_rel": crown_base_rel,
            "_crown_height": crown_height,
            "_crown_diameter_src": crown.get("source"),
            "_crown_diameter_mode": crown.get("mode"),
            "_trunk_diameter_cm": trunk_diameter_cm,
            "_trunk_radius_m": trunk_radius_m,
            "_trunk_diameter_src": crown.get("trunkSource") or trunk_info.get("source") or "—",
            "_tree_height_src": h_info.get("source"),
            "_idx": idx,
            "_kind": "tree",
        })
        return {"type": "Feature", "geometry": mapping(crown_geom), "properties": props}
    except Exception:
        return None


def normalize_trees(trees: dict[str, Any], route_buffer) -> dict[str, Any]:
    """HTML-faithful port of filterTreesInBuffer()/treeFeatureToCrown().

    Output `features` are crown geometries. Point/MultiPoint/LineString inputs
    are converted to circular crowns; Polygon/MultiPolygon inputs retain their
    original crown footprint. Trunk radius/diameter and vertical crown limits are
    stored in properties and used by the GNSS vegetation ray-tracing.
    """
    out: list[dict[str, Any]] = []
    bw, bs, be, bn = route_buffer.bounds
    # Cheap centre pre-filter before building the (relatively costly) crown geometry
    # and parsing every tree attribute. Crown radii are capped well under this margin
    # (~0.001 deg ~ 80-110 m here vs <=40 m max crown radius), so any tree whose crown
    # could touch the buffer survives and is still tested with the exact intersects()
    # below; the accepted set is identical to the all-trees scan, just ~80x faster on
    # a city-wide tree dataset.
    margin = 0.001
    for f in trees.get("features", []):
        geom_dict = f.get("geometry")
        if not geom_dict:
            continue
        try:
            clon, clat = _tree_center_lonlat(geom_dict)
            if clon is None or clon < bw - margin or clon > be + margin or clat < bs - margin or clat > bn + margin:
                continue
            crown = _tree_feature_to_crown(f, len(out))
            if not crown:
                continue
            geom = shape(crown["geometry"])
            if geom.is_empty:
                continue
            minx, miny, maxx, maxy = geom.bounds
            if maxx < bw or be < minx or maxy < bs or bn < miny:
                continue
            if geom.intersects(route_buffer):
                out.append(crown)
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": out}


def _turf_centroid_lonlat(geom_dict: dict[str, Any]) -> tuple[float, float] | None:
    """Match turf.centroid(): arithmetic mean of vertices, skipping each ring's
    closing coordinate (coordEach excludeWrapCoord). Differs from the shapely
    area centroid by metres on irregular footprints, which changes the DTM cell
    sampled for _base."""
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates")
    pts: list[tuple[float, float]] = []
    if gtype == "Point":
        pts = [tuple(coords[:2])]
    elif gtype in ("MultiPoint", "LineString"):
        pts = [tuple(c[:2]) for c in coords]
    elif gtype == "MultiLineString":
        pts = [tuple(c[:2]) for line in coords for c in line]
    elif gtype == "Polygon":
        pts = [tuple(c[:2]) for ring in coords for c in ring[:-1]]
    elif gtype == "MultiPolygon":
        pts = [tuple(c[:2]) for poly in coords for ring in poly for c in ring[:-1]]
    else:
        return None
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _tree_center_lonlat(geom_dict: dict[str, Any]) -> tuple[float | None, float | None]:
    """Cheap approximate centre of a tree feature for the spatial pre-filter."""
    gtype = geom_dict.get("type")
    coords = geom_dict.get("coordinates")
    try:
        if gtype == "Point":
            return float(coords[0]), float(coords[1])
        if gtype == "MultiPoint":
            return float(coords[0][0]), float(coords[0][1])
        cen = shape(geom_dict).centroid
        return float(cen.x), float(cen.y)
    except Exception:
        return None, None


def apply_dem_to_features(route: LineString, dem: DemModel, buildings: dict[str, Any], trees: dict[str, Any]) -> dict[str, int | float]:
    coords = list(route.coords)
    route_samples = [dem.sample(lon, lat) for lon, lat in coords]
    valid = [z for z in route_samples if math.isfinite(z)]
    if len(valid) < max(2, int(0.2 * len(coords))):
        raise ValueError("DTM does not cover enough of the trajectory. Processing stopped.")
    route_base = float(np.nanmean(valid))
    dem.route_base_m = route_base
    for coll in (buildings, trees):
        for f in coll.get("features", []):
            c = _turf_centroid_lonlat(f["geometry"])
            z = dem.sample(c[0], c[1]) if c else float("nan")
            props = f.setdefault("properties", {})
            if math.isfinite(z):
                props["terrain_z_m"] = z
                props["_base"] = z - route_base
            else:
                props["_base"] = 0.0
    return {"route_dem_samples": len(valid), "route_base_m": route_base, "buildings": len(buildings.get("features", [])), "trees": len(trees.get("features", []))}


def assert_route_inside_dem(route: LineString, dem: DemModel) -> None:
    coords = list(route.coords)
    samples = [dem.sample(lon, lat) for lon, lat in coords]
    ok = sum(math.isfinite(v) for v in samples)
    if ok < max(2, int(0.8 * len(coords))):
        raise ValueError(f"DTM coverage insufficient along route: {ok}/{len(coords)} route vertices sampled.")


def _overpass_query(buffer_bounds: tuple[float, float, float, float], timeout_s: int) -> str:
    w, s, e, n = buffer_bounds
    road_re = "^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|road|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link)$"
    return f"""
[out:json][timeout:{timeout_s}];
(
  way["highway"~"{road_re}"]["tunnel"]["tunnel"!~"^(no|false|0|culvert|canal|avalanche_protector|building_passage|arcade|passage)$"]({s},{w},{n},{e});
  way["highway"~"{road_re}"]["covered"]["covered"!~"^(no|false|0|arcade|colonnade|portico|porch|building_passage)$"]({s},{w},{n},{e});
  way["highway"~"{road_re}"]["bridge"]["bridge"!~"^(no|false|0)$"]({s},{w},{n},{e});
  way["railway"~"^(rail|light_rail|tram)$"]["bridge"]["bridge"!~"^(no|false|0)$"]({s},{w},{n},{e});
);
out tags geom;"""


# Overpass rejects the default "python-requests/x.y" User-Agent with HTTP 406
# (content negotiation) on overpass-api.de, which previously forced a fall-through
# to the slower mirror endpoints and ended in a read timeout with zero tunnels.
# A browser sends a real User-Agent/Accept pair, so we replicate that here to make
# the Python core behave like the WebUI fetch().
_OVERPASS_HEADERS = {
    "User-Agent": "AetherMMS/2.0 (GPS Solutions toolbox; GNSS mission planning)",
    "Accept": "application/json, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
}


MMS_DRIVABLE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "service",
    "living_street", "road", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
MMS_EXCLUDED_HIGHWAYS = {"footway", "path", "pedestrian", "cycleway", "steps", "corridor", "bridleway", "platform", "construction", "proposed"}


def _osm_tag_value(props: dict[str, Any], key: str) -> str:
    if not props:
        return ""
    return str(props.get(key, props.get(key.upper(), ""))).lower().strip()


def _osm_tag_is_yes(props: dict[str, Any], key: str) -> bool:
    value = _osm_tag_value(props, key)
    return bool(value) and value not in {"no", "false", "0", "none"}


def _is_mms_drivable_highway(props: dict[str, Any]) -> bool:
    highway = _osm_tag_value(props, "highway")
    return bool(highway) and highway not in MMS_EXCLUDED_HIGHWAYS and highway in MMS_DRIVABLE_HIGHWAYS


def _is_mms_overhead_rail(props: dict[str, Any]) -> bool:
    return _osm_tag_value(props, "railway") in {"rail", "light_rail", "tram"}


def _is_mms_excluded_covered_case(props: dict[str, Any]) -> bool:
    tunnel = _osm_tag_value(props, "tunnel")
    covered = _osm_tag_value(props, "covered")
    highway = _osm_tag_value(props, "highway")
    indoor = _osm_tag_value(props, "indoor")
    area = _osm_tag_value(props, "area")
    location = _osm_tag_value(props, "location")
    layer = _osm_tag_value(props, "layer")
    access = _osm_tag_value(props, "access")
    service = _osm_tag_value(props, "service")

    if tunnel in {"building_passage", "arcade", "passage", "yes;building_passage"}:
        return True
    if covered in {"arcade", "colonnade", "portico", "porch", "building_passage"}:
        return True
    if indoor == "yes" or area == "yes":
        return True
    if location == "underground" and not _is_mms_drivable_highway(props):
        return True
    try:
        if layer and float(layer) < 0 and not _is_mms_drivable_highway(props):
            return True
    except ValueError:
        pass
    if access == "private" and not _is_mms_drivable_highway(props):
        return True
    if service in {"parking_aisle", "driveway"} and not _osm_tag_is_yes(props, "tunnel") and not _osm_tag_is_yes(props, "bridge"):
        return True
    return highway in MMS_EXCLUDED_HIGHWAYS


def tunnel_semantic_score(props: dict[str, Any]) -> int:
    """Same MMS-oriented OSM semantic filter used by the HTML WebUI."""
    if not props or _is_mms_excluded_covered_case(props) or props.get("waterway") or props.get("WATERWAY"):
        return 0

    tunnel = _osm_tag_value(props, "tunnel")
    is_tunnel = _osm_tag_is_yes(props, "tunnel") and tunnel not in {"culvert", "canal", "avalanche_protector"}
    is_covered = _osm_tag_is_yes(props, "covered")
    is_bridge = _osm_tag_is_yes(props, "bridge")
    drivable = _is_mms_drivable_highway(props)
    overhead_rail = _is_mms_overhead_rail(props)

    if (is_tunnel or is_covered) and not drivable:
        return 0
    if is_bridge and not (drivable or overhead_rail):
        return 0
    if not is_tunnel and not is_covered and not is_bridge:
        return 0

    score = 0
    if is_tunnel:
        score += 5
    if is_covered:
        score += 3
    if is_bridge:
        score += 3
    if drivable:
        score += 3
    if overhead_rail:
        score += 2
    if is_bridge and drivable:
        score += 1
    return score


def _feature_metric(feature: dict[str, Any], fwd: Transformer):
    return to_metric(shape(feature["geometry"]), fwd)


def _min_distance_feature_to_route_m(feature: dict[str, Any], route: LineString, fwd: Transformer | None = None) -> float:
    try:
        fwd = fwd or transformers(route)[0]
        return float(_feature_metric(feature, fwd).distance(to_metric(route, fwd)))
    except Exception:
        return float("inf")


def _feature_length_m(feature: dict[str, Any], route: LineString, fwd: Transformer | None = None) -> float:
    try:
        fwd = fwd or transformers(route)[0]
        geom_m = _feature_metric(feature, fwd)
        if geom_m.geom_type in {"Polygon", "MultiPolygon"}:
            return math.sqrt(max(0.0, float(geom_m.area)))
        return float(geom_m.length)
    except Exception:
        return 0.0


def _estimate_route_tunnel_overlap_m(feature: dict[str, Any], route: LineString, tolerance_m: float, fwd: Transformer | None = None) -> float:
    try:
        fwd = fwd or transformers(route)[0]
        geom_m = _feature_metric(feature, fwd)
        route_m = to_metric(route, fwd)
        tunnel_mask = geom_m if geom_m.geom_type in {"Polygon", "MultiPolygon"} else geom_m.buffer(tolerance_m)
        if route_m.length <= 0:
            return 0.0
        step_m = 3.0
        overlap = 0.0
        prev_inside = False
        d = 0.0
        while d <= route_m.length:
            inside = tunnel_mask.contains(route_m.interpolate(d)) or tunnel_mask.touches(route_m.interpolate(d))
            if inside or prev_inside:
                overlap += step_m
            prev_inside = inside
            d += step_m
        return overlap
    except Exception:
        return 0.0


def _bearing_diff_deg(a: float, b: float) -> float:
    d = abs((((a - b) % 360.0) + 540.0) % 360.0 - 180.0)
    if d > 90.0:
        d = 180.0 - d
    return abs(d)


def _flatten_line_coords(geom) -> list[tuple[float, float]]:
    if geom.geom_type == "LineString":
        return [(float(x), float(y)) for x, y in geom.coords]
    if geom.geom_type == "MultiLineString":
        coords: list[tuple[float, float]] = []
        for line in geom.geoms:
            coords.extend((float(x), float(y)) for x, y in line.coords)
        return coords
    return []


def _line_bearing_at_middle(feature: dict[str, Any]) -> float | None:
    try:
        coords = _flatten_line_coords(shape(feature["geometry"]))
        if len(coords) < 2:
            return None
        line = LineString(coords)
        length = turf_route_length_m(line)
        if not math.isfinite(length) or length <= 0:
            return None
        a = turf_route_position_at_distance(line, max(0.0, length * 0.45))
        b = turf_route_position_at_distance(line, min(length, length * 0.55))
        return _turf_bearing_deg(a, b)
    except Exception:
        return None


def _route_bearing_near_feature(route: LineString, feature: dict[str, Any]) -> float | None:
    try:
        route_len_m = turf_route_length_m(route)
        if not math.isfinite(route_len_m) or route_len_m <= 0:
            return None
        center = shape(feature["geometry"]).centroid
        step_m = max(5.0, min(30.0, route_len_m / 120.0))
        best_d = 0.0
        best_dist = float("inf")
        d = 0.0
        while d <= route_len_m:
            lon, lat = turf_route_position_at_distance(route, d)
            dm = _turf_distance_m((center.x, center.y), (lon, lat))
            if dm < best_dist:
                best_dist = dm
                best_d = d
            d += step_m
        d1 = max(0.0, best_d - 15.0)
        d2 = min(route_len_m, best_d + 15.0)
        if d2 <= d1:
            return None
        a = turf_route_position_at_distance(route, d1)
        b = turf_route_position_at_distance(route, d2)
        return _turf_bearing_deg(a, b)
    except Exception:
        return None


def tunnel_actually_crosses_route(feature: dict[str, Any], route: LineString, fwd: Transformer | None = None) -> bool:
    """Equivalent to the HTML MMS geometry filter: accept only route-crossing/nearby tunnel lines."""
    try:
        props = feature.get("properties") or {}
        is_bridge = _osm_tag_is_yes(props, "bridge") and not _osm_tag_is_yes(props, "tunnel") and not _osm_tag_is_yes(props, "covered")
        is_tunnel_or_covered = _osm_tag_is_yes(props, "tunnel") or _osm_tag_is_yes(props, "covered")
        if not is_bridge and not is_tunnel_or_covered:
            return False
        if tunnel_semantic_score(props) <= 0:
            return False

        max_axis_distance_m = 7.0 if is_bridge else 8.0
        min_length_m = 6.0 if is_bridge else 8.0
        min_overlap_m = 4.0 if is_bridge else 8.0
        fwd = fwd or transformers(route)[0]

        if _feature_length_m(feature, route, fwd) < min_length_m:
            return False
        d = _min_distance_feature_to_route_m(feature, route, fwd)
        if not math.isfinite(d) or d > max_axis_distance_m:
            return False
        overlap = _estimate_route_tunnel_overlap_m(feature, route, max_axis_distance_m, fwd)
        if overlap < min_overlap_m:
            return False

        geom_type = shape(feature["geometry"]).geom_type
        if geom_type in {"LineString", "MultiLineString"}:
            tb = _line_bearing_at_middle(feature)
            rb = _route_bearing_near_feature(route, feature)
            if tb is not None and rb is not None and math.isfinite(tb) and math.isfinite(rb):
                diff = _bearing_diff_deg(tb, rb)
                if is_bridge and diff < 45.0:
                    return False
                if not is_bridge and diff > 35.0:
                    return False
                props["_bearing_diff_deg"] = diff
                props["_route_overlap_m"] = overlap
                props["_mms_structure"] = "overpass" if is_bridge else "tunnel_or_covered_road"
                feature["properties"] = props
        return True
    except Exception:
        return False


def _write_osm_cache(cache_file: Path | None, fc: dict[str, Any]) -> None:
    """Write the single OSM tunnel cache for this run (no timestamped snapshots)."""
    if not cache_file:
        return
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    except Exception:
        pass


def _osm_status(message: str) -> None:
    """Single concise OpenStreetMap status line, aligned with the report block."""
    print(f"[AetherMMS]       OpenStreetMap: {message}")


def fetch_osm_tunnels(route: LineString, route_buffer, cache_file: Path | None, use_cache: bool, endpoints: list[str], timeout_s: int = 25) -> dict[str, Any]:
    if use_cache and cache_file and cache_file.exists():
        cached_fc = load_geojson(cache_file)
        n_cached = len(cached_fc.get("features", []))
        _osm_status(f"OK (cache, {n_cached} segment{'s' if n_cached != 1 else ''})")
        return cached_fc
    query_timeout_s = max(3, int(timeout_s))
    # The WebUI fetch() has no client-side read timeout and simply waits for the
    # server to honour the Overpass [timeout:N] directive. Mirror that patience so
    # a momentarily queued endpoint is not abandoned prematurely.
    request_timeout = (10, max(60, query_timeout_s + 30))
    query = _overpass_query(route_buffer.bounds, query_timeout_s)
    last_err: Exception | None = None
    saw_live_zero = False
    live_endpoints = list(endpoints or [])
    for endpoint in live_endpoints:
        for mode in ("POST", "GET"):
            try:
                if mode == "POST":
                    res = requests.post(endpoint, data={"data": query}, timeout=request_timeout, headers=_OVERPASS_HEADERS)
                else:
                    res = requests.get(endpoint, params={"data": query}, timeout=request_timeout, headers=_OVERPASS_HEADERS)
                res.raise_for_status()
                raw_fc = _overpass_raw_to_geojson(res.json())
                if not raw_fc.get("features"):
                    saw_live_zero = True
                    continue
                fc = _filter_osm_tunnels(raw_fc, route, route_buffer)
                n_live = len(fc.get("features", []))
                if n_live:
                    _write_osm_cache(cache_file, fc)
                    _osm_status(f"OK ({n_live} tunnel/covered/overpass segment{'s' if n_live != 1 else ''})")
                    return fc
                saw_live_zero = True
            except Exception as exc:
                last_err = exc
    if saw_live_zero:
        fc = {"type": "FeatureCollection", "features": []}
        _write_osm_cache(cache_file, fc)
        _osm_status("OK (no tunnels in area)")
        return fc
    reason = type(last_err).__name__ if last_err else "no endpoint configured"
    _osm_status(f"unavailable ({reason}) - continuing without tunnels")
    return {"type": "FeatureCollection", "features": []}


def _overpass_raw_to_geojson(data: dict[str, Any]) -> dict[str, Any]:
    features = []
    for el in data.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        coords = [(float(p["lon"]), float(p["lat"])) for p in geom if "lon" in p and "lat" in p]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        line = LineString(coords)
        props = {"osm_id": el.get("id"), "osm_type": el.get("type", "way"), "_source": "OpenStreetMap/Overpass", **tags}
        features.append({"type": "Feature", "geometry": mapping(line), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _filter_osm_tunnels(raw_fc: dict[str, Any], route: LineString, route_buffer) -> dict[str, Any]:
    features = []
    fwd, inv = transformers(route)
    route_m = to_metric(route, fwd)
    for idx, feature in enumerate(raw_fc.get("features", []), start=1):
        try:
            line = shape(feature["geometry"])
        except Exception:
            continue
        props0 = feature.get("properties") or {}
        if tunnel_semantic_score(props0) <= 0:
            continue
        if not line.intersects(route_buffer):
            continue
        if not tunnel_actually_crosses_route(feature, route, fwd):
            continue
        dist = route_m.distance(to_metric(line, fwd))
        props = dict(feature.get("properties") or {})
        props.update({
            "_tunnel_idx": idx,
            "_tunnel_name": props.get("name") or props.get("Name") or props.get("nome") or props.get("NOME") or props.get("ref") or f"Tunnel {idx}",
            "_tunnel_score": tunnel_semantic_score(props),
            "_route_distance_m": dist,
        })
        try:
            if line.geom_type in {"Polygon", "MultiPolygon"}:
                norm_geom = line
                props["_tunnel_geom"] = "area"
            else:
                norm_geom = shp_transform(inv.transform, to_metric(line, fwd).buffer(5.0))
                props["_tunnel_geom"] = "line-buffer-5m"
        except Exception:
            norm_geom = line
            props["_tunnel_geom"] = "line"
        features.append({"type": "Feature", "geometry": mapping(norm_geom), "properties": props})
    return {"type": "FeatureCollection", "features": features}


def _overpass_to_geojson(data: dict[str, Any], route: LineString, route_buffer) -> dict[str, Any]:
    return _filter_osm_tunnels(_overpass_raw_to_geojson(data), route, route_buffer)


def _make_metric_scene(route: LineString, buildings: dict[str, Any], trees: dict[str, Any], tunnels: dict[str, Any]) -> MetricScene:
    fwd, inv = transformers(route)
    route_m = to_metric(route, fwd)
    building_geoms = []
    building_lonlat_geoms = []
    building_props = []
    for f in buildings.get("features", []):
        try:
            g_ll = shape(f["geometry"])
            g = to_metric(g_ll, fwd)
            if not g.is_empty:
                building_geoms.append(g)
                building_lonlat_geoms.append(g_ll)
                building_props.append(dict(f.get("properties") or {}))
        except Exception:
            continue
    tunnel_geoms = []
    for f in tunnels.get("features", []):
        try:
            g = to_metric(shape(f["geometry"]), fwd)
            if not g.is_empty:
                tunnel_geoms.append(g)
        except Exception:
            continue
    tree_props = []
    for f in trees.get("features", []):
        p = dict(f.get("properties") or {})
        try:
            lon = float(p.get("_tree_center_lon")); lat = float(p.get("_tree_center_lat"))
            x, y = fwd.transform(lon, lat)
            p["_x_m"] = x; p["_y_m"] = y
            tree_props.append(p)
        except Exception:
            continue
    return MetricScene(
        fwd=fwd,
        inv=inv,
        route_m=route_m,
        building_geoms=building_geoms,
        building_lonlat_geoms=building_lonlat_geoms,
        building_props=building_props,
        building_tree=STRtree(building_geoms) if building_geoms else None,
        tree_props=tree_props,
        tunnel_geoms=tunnel_geoms,
        tunnel_tree=STRtree(tunnel_geoms) if tunnel_geoms else None,
    )


def is_position_in_tunnel(point: Point, tunnels: dict[str, Any], scene: MetricScene | None = None) -> bool:
    """Faithful port of the HTML isPositionInTunnel(): strict point-in-polygon.

    The stored tunnel features are already a 5 m buffer around the tunnel/covered
    centreline (or the original area polygon), so the WebUI tests the position with
    turf.booleanPointInPolygon (boundary counts as inside). We replicate that with
    shapely covers(); a distance tolerance would flag far more epochs as tunnel.
    """
    if scene and scene.tunnel_tree:
        x, y = scene.fwd.transform(point.x, point.y)
        p_m = Point(x, y)
        for idx in scene.tunnel_tree.query(p_m):
            try:
                if scene.tunnel_geoms[int(idx)].covers(p_m):
                    return True
            except Exception:
                continue
        return False
    if not tunnels.get("features"):
        return False
    p = Point(point.x, point.y)
    for f in tunnels.get("features", []):
        try:
            if shape(f["geometry"]).covers(p):
                return True
        except Exception:
            continue
    return False


def evaluate_base_coverage(route: LineString, bases: list[Point], step_m: float = 100.0) -> dict[str, Any]:
    """HTML-faithful base check: sample route every 0.1 km and test nearest base.

    Status thresholds are the same as evaluateBaseCoverage() in the HTML app:
    green <= 5 km, yellow <= 10 km, orange <= 15 km, red > 15 km.
    """
    if not bases:
        return {"status": "none", "color": "#5a6f91", "max_km": None, "label": "Bases: not loaded"}

    fwd, _ = transformers(route)
    route_m = to_metric(route, fwd)
    base_m = [to_metric(b, fwd) for b in bases]
    samples = []
    n = max(1, int(math.ceil(route_m.length / max(1.0, step_m))))
    for i in range(n + 1):
        samples.append(route_m.interpolate(min(route_m.length, i * step_m)))
    for lon, lat in route.coords:
        samples.append(to_metric(Point(lon, lat), fwd))

    max_km = 0.0
    for p in samples:
        min_m = min((p.distance(b) for b in base_m), default=float("inf"))
        if math.isfinite(min_m):
            max_km = max(max_km, min_m / 1000.0)

    if max_km <= 5:
        return {"status": "green", "color": "#00e676", "max_km": max_km, "label": f"OK - entire trajectory within 5 km of the bases (max {max_km:.2f} km)"}
    if max_km <= 10:
        return {"status": "yellow", "color": "#ffc107", "max_km": max_km, "label": f"Warning - trajectory between 5 and 10 km from the bases (max {max_km:.2f} km)"}
    if max_km <= 15:
        return {"status": "orange", "color": "#ff9800", "max_km": max_km, "label": f"Critical - trajectory between 10 and 15 km from the bases (max {max_km:.2f} km)"}
    return {"status": "red", "color": "#f44336", "max_km": max_km, "label": f"Out of range - trajectory more than 15 km from the bases (max {max_km:.2f} km)"}


def _resolve_path(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else base / p


def build_city(config: dict[str, Any], root: Path) -> dict[str, Any]:
    example_dir = _resolve_path(root, config["data_dir"]) or (root / "Data")
    inputs = config["inputs"]
    dem = load_dem(_resolve_path(example_dir, inputs["dem_geotiff"]) or example_dir / "DTM5x5.tif")
    route = extract_longest_route(_resolve_path(example_dir, inputs["trajectory_kml"]) or example_dir / "Traj.kml")
    assert_route_inside_dem(route, dem)
    buffer_m = float(config.get("mms", {}).get("lidar_range_m", 100.0))
    route_buffer = buffer_route(route, buffer_m)
    buildings = normalize_buildings(load_geojson(_resolve_path(example_dir, inputs.get("buildings_geojson")), required=True), route_buffer)
    trees = normalize_trees(load_geojson(_resolve_path(example_dir, inputs.get("trees_geojson")), required=False), route_buffer)
    dem_stats = apply_dem_to_features(route, dem, buildings, trees)
    osm_cfg = config.get("osm", {})
    tunnels = {"type": "FeatureCollection", "features": []}
    if osm_cfg.get("enabled", False):
        cache_name = osm_cfg.get("cache_file") or "osm_tunnels_cache.geojson"
        tunnels = fetch_osm_tunnels(route, route_buffer, _resolve_path(example_dir, cache_name), bool(osm_cfg.get("use_cache", True)), list(osm_cfg.get("endpoints", [])), int(osm_cfg.get("overpass_timeout_s", 25)))
    bases_path = _resolve_path(example_dir, inputs.get("bases_kml"))
    bases = parse_kml_points(bases_path)
    base_coverage = evaluate_base_coverage(route, bases)
    scene = _make_metric_scene(route, buildings, trees, tunnels)
    return {"dem": dem, "route": route, "route_buffer": route_buffer, "buildings": buildings, "trees": trees, "tunnels": tunnels, "bases": bases, "base_coverage": base_coverage, "dem_stats": dem_stats, "route_length_m": route_length_m(route), "scene": scene, "example_dir": example_dir}
