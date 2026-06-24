"""AetherMMS GNSS visibility and temporal-window prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString, Point

GEOD = Geod(ellps="WGS84")

try:
    from .buildingcity import MetricScene, is_position_in_tunnel, turf_route_position_at_distance, _turf_destination, _turf_distance_m, _turf_bearing_deg
    from .satellitepropagation import Satellite, satellite_ecef, look_angles_from_sat
except ImportError:
    from buildingcity import MetricScene, is_position_in_tunnel, turf_route_position_at_distance, _turf_destination, _turf_distance_m, _turf_bearing_deg
    from satellitepropagation import Satellite, satellite_ecef, look_angles_from_sat


@dataclass
class RayBlock:
    blocked: bool = False
    affected: bool = False
    kind: str | None = None
    dist_m: float | None = None
    reason: str | None = None


def _dt(date_s: str, time_s: str) -> datetime:
    if len(time_s.split(":")) == 2:
        time_s += ":00"
    return datetime.fromisoformat(f"{date_s}T{time_s}").replace(tzinfo=timezone.utc)


def _format_time(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%H:%M:%S")


def route_position_at_distance(route: LineString, distance_m: float, scene: MetricScene | None = None) -> tuple[float, float]:
    """Equivalent to Turf `along(route, distM/1000)` used by the WebUI."""
    return turf_route_position_at_distance(route, distance_m)


def _pdop(looks: list[dict[str, Any]]) -> float:
    # Port of computePdopFromLooks()
    if len(looks) < 4:
        return float("nan")
    G = []
    for l in looks:
        az = math.radians(l["az"]); el = math.radians(l["el"])
        ux = math.cos(el) * math.sin(az)
        uy = math.cos(el) * math.cos(az)
        uz = math.sin(el)
        G.append([ux, uy, uz, 1.0])
    try:
        a = np.asarray(G, dtype=float)
        q = np.linalg.inv(a.T @ a)
        pdop = math.sqrt(max(0.0, q[0, 0] + q[1, 1] + q[2, 2]))
        return pdop if math.isfinite(pdop) and pdop < 999 else float("nan")
    except Exception:
        return float("nan")


def _finite_mean(vals: list[float]) -> float:
    vv = [float(v) for v in vals if isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))]
    return float(np.mean(vv)) if vv else float("nan")


def _safe_mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _constellation(prn: str) -> str:
    return "GPS" if str(prn).startswith("G") else "Galileo" if str(prn).startswith("E") else "UNKNOWN"


def _destination_lonlat(lon: float, lat: float, az_deg: float, dist_m: float) -> tuple[float, float]:
    """Spherical destination equivalent to Turf.destination()."""
    return _turf_destination((lon, lat), dist_m, az_deg)


def _first_points_from_intersection(geom, origin_m: Point) -> list[tuple[float, Point]]:
    """Extract candidate entry points from a shapely ray/footprint intersection."""
    if geom.is_empty:
        return []
    pts: list[Point] = []
    gt = geom.geom_type
    if gt == "Point":
        pts.append(geom)
    elif gt == "MultiPoint":
        pts.extend(list(geom.geoms))
    elif gt in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        if coords:
            pts.append(Point(coords[0])); pts.append(Point(coords[-1]))
    elif gt == "MultiLineString":
        for g in geom.geoms:
            coords = list(g.coords)
            if coords:
                pts.append(Point(coords[0])); pts.append(Point(coords[-1]))
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            pts.extend([p for _, p in _first_points_from_intersection(g, origin_m)])
    out = []
    for p in pts:
        d = origin_m.distance(p)
        if d >= 0.25 and math.isfinite(d):
            out.append((d, p))
    out.sort(key=lambda x: x[0])
    return out


def _intersection_points(geom) -> list[Point]:
    if geom.is_empty:
        return []
    pts: list[Point] = []
    gt = geom.geom_type
    if gt == "Point":
        pts.append(geom)
    elif gt == "MultiPoint":
        pts.extend(list(geom.geoms))
    elif gt in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        if coords:
            pts.append(Point(coords[0])); pts.append(Point(coords[-1]))
    elif gt == "MultiLineString":
        for g in geom.geoms:
            coords = list(g.coords)
            if coords:
                pts.append(Point(coords[0])); pts.append(Point(coords[-1]))
    elif gt == "GeometryCollection":
        for g in geom.geoms:
            pts.extend(_intersection_points(g))
    return pts


def is_ray_blocked_by_buildings(lon: float, lat: float, antenna_h: float, az: float, el: float, scene: MetricScene, max_km: float = 1.2) -> RayBlock:
    """Port of HTML isRayBlockedByBuildings()."""
    if not scene.building_tree or el <= 0 or not math.isfinite(el):
        return RayBlock(blocked=False)
    ox, oy = scene.fwd.transform(lon, lat)
    origin = Point(ox, oy)
    origin_ll = Point(lon, lat)
    azr = math.radians(az)
    max_m = max_km * 1000.0
    dest = Point(ox + max_m * math.sin(azr), oy + max_m * math.cos(azr))
    ray = LineString([origin, dest])
    ray_ll = LineString([(lon, lat), _turf_destination((lon, lat), max_m, az)])
    best: RayBlock | None = None
    # STRtree gives the same role as Turf bbox prefilter, but faster.
    for idx in scene.building_tree.query(ray):
        i = int(idx)
        geom = scene.building_lonlat_geoms[i]
        props = scene.building_props[i]
        try:
            b_height = max(0.0, float(props.get("_height", props.get("height_m", 16.0))))
            b_base = max(0.0, float(props.get("_base", props.get("base_relative_to_route_m", 0.0))))
            roof_z = b_base + b_height
            if roof_z <= antenna_h:
                continue
            if geom.contains(origin_ll):
                return RayBlock(blocked=True, kind="building", dist_m=0.0, reason="antenna-inside-footprint")
            if not geom.intersects(ray_ll):
                continue
            inter = ray_ll.intersection(geom)
            points = []
            for p in _intersection_points(inter):
                dist_m = _turf_distance_m((lon, lat), (p.x, p.y))
                if dist_m >= 0.25 and math.isfinite(dist_m):
                    points.append((dist_m, p))
            points.sort(key=lambda x: x[0])
            for dist_m, _p in points:
                ray_z = antenna_h + math.tan(math.radians(el)) * dist_m
                if ray_z <= roof_z:
                    cand = RayBlock(blocked=True, kind="building", dist_m=dist_m, reason="ray-intersects-building-footprint")
                    if best is None or (cand.dist_m or 0) < (best.dist_m or 1e99):
                        best = cand
                    break
        except Exception:
            continue
    return best or RayBlock(blocked=False)


def is_ray_blocked_by_dem(lon: float, lat: float, antenna_h: float, az: float, el: float, dem, max_km: float = 1.2) -> RayBlock:
    """Terrain obstruction: the DTM profile is sampled along the ray azimuth at
    the DTM ground resolution, so the step adapts to the input DTM."""
    if el <= 0 or not math.isfinite(el):
        return RayBlock(blocked=False)
    if not dem.contains_lonlat(lon, lat):
        return RayBlock(blocked=False, reason="origin-outside-dem")
    origin_terrain = dem.sample_relative(lon, lat)
    if not math.isfinite(origin_terrain):
        return RayBlock(blocked=False, reason="origin-dem-nodata")
    tan_el = math.tan(math.radians(el))
    max_m = max(50.0, max_km * 1000.0)
    step_m = max(1.0, float(getattr(dem, "res_m", 5.0)))
    antenna_abs = origin_terrain + antenna_h
    # Same loop bound as the JS for-loop (no epsilon): both accumulate d with
    # identical IEEE additions, so the sampled profile distances stay in sync.
    d = step_m
    while d <= max_m:
        xlon, xlat = _destination_lonlat(lon, lat, az, d)
        terrain = dem.sample_relative(xlon, xlat)
        if math.isfinite(terrain):
            ray_z = antenna_abs + tan_el * d
            clearance = ray_z - terrain
            if clearance < 0.50:
                return RayBlock(blocked=True, kind="terrain", dist_m=d, reason="dem-terrain")
        d += step_m
    return RayBlock(blocked=False)


def is_ray_affected_by_trees(lon: float, lat: float, antenna_h: float, az: float, el: float, scene: MetricScene, max_km: float = 1.2) -> RayBlock:
    """HTML-faithful port of isRayAffectedByTrees().

    Vegetation is modelled as two coaxial vertical cylinders exactly as in the
    JavaScript implementation:
    - crown cylinder: radius _crown_radius_m, z=[_base+_crown_base_rel, _base+_tree_height]
    - trunk cylinder: radius _trunk_radius_m, z=[_base, _base+_crown_base_rel]

    The horizontal geometry follows Turf's distance+bearing workflow rather than
    relying on tree polygon intersections: center distance, bearing difference,
    along-track coordinate and perpendicular offset define the cylinder hit.
    """
    if not scene.tree_props or el <= 0 or not math.isfinite(el):
        return RayBlock(affected=False)

    max_m = max_km * 1000.0
    tan_el = math.tan(math.radians(el))
    best: RayBlock | None = None

    for p in scene.tree_props:
        try:
            center_lon = float(p.get("_tree_center_lon"))
            center_lat = float(p.get("_tree_center_lat"))
            h = max(0.0, float(p.get("_tree_height", 0.0) or 0.0))
            base_abs = max(0.0, float(p.get("_base", 0.0) or 0.0))
            crown_radius = max(0.25, float(p.get("_crown_radius_m", float(p.get("_crown_diameter", 0.0) or 0.0) / 2.0) or 0.0))
            crown_base_abs = base_abs + max(0.0, float(p.get("_crown_base_rel", h * 0.38) or 0.0))
            crown_top_abs = base_abs + h
            trunk_radius = max(0.05, float(p.get("_trunk_radius_m", 0.12) or 0.12))

            if not all(math.isfinite(v) for v in (center_lon, center_lat)):
                continue
            if h <= 0.0 or crown_top_abs <= antenna_h:
                continue

            # Turf.distance + Turf.bearing equivalent.
            center_dist_m = _turf_distance_m((lon, lat), (center_lon, center_lat))
            if not math.isfinite(center_dist_m) or center_dist_m > max_m + crown_radius:
                continue

            bearing_to_center = _turf_bearing_deg((lon, lat), (center_lon, center_lat))
            diff_deg = ((bearing_to_center - az + 540.0) % 360.0) - 180.0
            diff = math.radians(diff_deg)
            along_m = center_dist_m * math.cos(diff)
            perp_m = abs(center_dist_m * math.sin(diff))
            if along_m < -crown_radius or along_m > max_m + crown_radius:
                continue

            def test_cylinder(radius: float, z_bot: float, z_top: float, reason: str) -> RayBlock | None:
                if perp_m > radius:
                    return None
                dx = math.sqrt(max(0.0, radius * radius - perp_m * perp_m))
                entry_m = max(0.0, along_m - dx)
                exit_m = min(max_m, along_m + dx)
                if exit_m < entry_m:
                    return None
                z_entry = antenna_h + tan_el * entry_m
                z_exit = antenna_h + tan_el * exit_m
                z_ray_bot = min(z_entry, z_exit)
                z_ray_top = max(z_entry, z_exit)
                if z_ray_top < z_bot or z_ray_bot > z_top:
                    return None
                return RayBlock(affected=True, kind="vegetation", dist_m=entry_m, reason=reason)

            for cand in (
                test_cylinder(crown_radius, crown_base_abs, crown_top_abs, "ray-intersects-crown-cylinder"),
                test_cylinder(trunk_radius, base_abs, crown_base_abs, "ray-intersects-trunk-cylinder"),
            ):
                if cand and (best is None or (cand.dist_m or 0.0) < (best.dist_m or 1e99)):
                    best = cand
        except Exception:
            continue

    return best or RayBlock(affected=False)


def classify_gnss_obstruction(building: RayBlock, tree: RayBlock, dem: RayBlock) -> tuple[str, str | None]:
    # Priority: DTM/building hard NLOS; vegetation possible degradation.
    hard = [b for b in (building, dem) if b and b.blocked]
    hard.sort(key=lambda b: b.dist_m if b.dist_m is not None else 0.0)
    if hard:
        return "NLOS", hard[0].kind
    if tree and tree.affected:
        return "VEG", tree.kind
    return "LOS", None


def score_survey_window(rows: list[dict[str, Any]]) -> dict[str, float]:
    # Direct port of scoreSurveyWindow().
    if not rows:
        return {"score": -math.inf, "avg_los": 0, "avg_svi": 0, "avg_pdop": math.inf, "outage_pct": 100, "score_outage_pct": 100, "tunnel_pct": 0, "los_score": 0, "pdop_score": 0, "continuity_score": 0}
    avg_los = _safe_mean([r.get("los", 0) for r in rows])
    avg_svi = _finite_mean([r.get("sviTotal", r.get("svi_total", float("nan"))) for r in rows])
    avg_pdop = _finite_mean([r.get("pdopLos", r.get("pdop_los_total", float("nan"))) for r in rows])
    tunnel_pct = 100.0 * sum(1 for r in rows if float(r.get("tunnel", 0) or 0) > 0) / len(rows)
    outage_pct = 100.0 * sum(1 for r in rows if float(r.get("tunnel", 0) or 0) > 0 or (r.get("los", 0) or 0) < 4 or not math.isfinite(float(r.get("pdopLos", r.get("pdop_los_total", float("nan"))) or float("nan"))) or float(r.get("pdopLos", r.get("pdop_los_total", 999)) or 999) > 8) / len(rows)
    score_rows = [r for r in rows if float(r.get("tunnel", 0) or 0) <= 0]
    ranking = score_rows or rows
    score_avg_los = _safe_mean([r.get("los", 0) for r in ranking])
    score_avg_pdop = _finite_mean([r.get("pdopLos", r.get("pdop_los_total", float("nan"))) for r in ranking])
    score_outage_pct = 100.0 * sum(1 for r in ranking if (r.get("los", 0) or 0) < 4 or not math.isfinite(float(r.get("pdopLos", r.get("pdop_los_total", float("nan"))) or float("nan"))) or float(r.get("pdopLos", r.get("pdop_los_total", 999)) or 999) > 8) / len(ranking)
    los_score = max(0.0, min(1.0, score_avg_los / 10.0))
    pdop_score = max(0.0, min(1.0, (8.0 - score_avg_pdop) / 6.0)) if math.isfinite(score_avg_pdop) else 0.0
    continuity_score = max(0.0, 1.0 - score_outage_pct / 100.0)
    score = 100.0 * (0.40 * los_score + 0.25 * pdop_score + 0.35 * continuity_score)
    return {"score": score, "avg_los": avg_los, "avg_svi": avg_svi, "avg_pdop": avg_pdop, "outage_pct": outage_pct, "score_outage_pct": score_outage_pct, "tunnel_pct": tunnel_pct, "los_score": los_score, "pdop_score": pdop_score, "continuity_score": continuity_score}


def simulate_mission(city: dict[str, Any], satellites: list[Satellite], config: dict[str, Any], start_dt: datetime | None = None, quick: bool = False, max_epochs: int | None = None, skip_looks: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Port of runGnssPlanningAt1Hz().

    `quick=True` is used by temporal planning preview/refinement to mimic the
    HTML option that skips vegetation for the fast daily preview.
    """
    route = city["route"]; dem = city["dem"]; scene: MetricScene = city["scene"]
    route_len = city["route_length_m"]
    speed_mps = float(config["survey"].get("speed_kmh", 30.0)) / 3.6
    antenna_h = float(config.get("mms", {}).get("antenna_height_m", 2.2))
    elev_mask = float(config["survey"].get("elevation_mask_deg", 10.0))
    # HTML hard-codes 1.2 km for DTM/building/tree ray tests in GNSS analysis.
    ray_max_km = float(config.get("ray_tracing", {}).get("max_ray_km", 1.2))
    step_sec = int(config["survey"].get("step_sec", 1))
    if max_epochs is not None:
        # Port of simulateGnssRowsForStart(): temporal prediction uses a stride
        # so the preview/refinement is fast and does not run every 1-Hz epoch.
        tmp_total = max(0, int(math.floor(route_len / max(0.01, speed_mps))))
        step_sec = max(1, int(math.ceil((tmp_total + 1) / max(1, int(max_epochs)))))
    if start_dt is None:
        start_dt = _dt(config["survey"]["date_utc"], config["survey"]["start_time_utc"])
    total_sec = max(0, int(math.floor(route_len / max(0.01, speed_mps))))
    epoch_rows: list[dict[str, Any]] = []
    look_rows: list[dict[str, Any]] = []

    # Precalculate epoch positions exactly as the HTML does.
    epoch_positions = []
    for t in range(0, total_sec + 1, step_sec):
        dist_m = min(route_len, speed_mps * t)
        lon, lat = route_position_at_distance(route, dist_m, scene)
        epoch_positions.append((t, dist_m, lon, lat))

    for t, dist_m, lon, lat in epoch_positions:
        epoch = start_dt + timedelta(seconds=t)
        in_tunnel = is_position_in_tunnel(Point(lon, lat), city.get("tunnels", {}), scene)
        visible = los = nlos = veg = 0
        epoch_looks: list[dict[str, Any]] = []
        for sat in satellites:
            ecef = satellite_ecef(sat, epoch)
            if ecef is None:
                continue
            az, el, rng = look_angles_from_sat(ecef, lon, lat, antenna_h)
            if el < elev_mask:
                continue
            visible += 1
            if in_tunnel:
                nlos += 1
                row = {"t_s": t, "dist_m": dist_m, "epoch_iso": epoch.isoformat(), "lon": lon, "lat": lat, "sv": sat.prn, "constellation": sat.constellation, "az": az, "el": el, "range_m": rng, "status": "GNSS_DENIED", "block_kind": "gnss_denied_tunnel_or_overpass"}
                if not skip_looks:
                    look_rows.append(row)
                epoch_looks.append(row)
                continue
            dem_block = is_ray_blocked_by_dem(lon, lat, antenna_h, az, el, dem, ray_max_km)
            building_block = is_ray_blocked_by_buildings(lon, lat, antenna_h, az, el, scene, ray_max_km)
            tree_effect = RayBlock(affected=False) if quick else is_ray_affected_by_trees(lon, lat, antenna_h, az, el, scene, ray_max_km)
            status, block_kind = classify_gnss_obstruction(building_block, tree_effect, dem_block)
            if status == "NLOS":
                nlos += 1
            elif status == "VEG":
                veg += 1
            else:
                los += 1
            row = {"t_s": t, "dist_m": dist_m, "epoch_iso": epoch.isoformat(), "lon": lon, "lat": lat, "sv": sat.prn, "constellation": sat.constellation, "az": az, "el": el, "range_m": rng, "status": status, "block_kind": block_kind}
            if not skip_looks:
                look_rows.append(row)
            epoch_looks.append(row)

        los_looks = [l for l in epoch_looks if l["status"] == "LOS"]
        gps_looks = [l for l in epoch_looks if _constellation(l["sv"]) == "GPS"]
        gal_looks = [l for l in epoch_looks if _constellation(l["sv"]) == "Galileo"]
        gps_los = [l for l in los_looks if _constellation(l["sv"]) == "GPS"]
        gal_los = [l for l in los_looks if _constellation(l["sv"]) == "Galileo"]
        pdop_present = _pdop(epoch_looks)
        pdop_los = _pdop(los_looks)
        pdop_gps = _pdop(gps_los)
        pdop_gal = _pdop(gal_los)
        svi_total = 0.0 if in_tunnel else (los / visible if visible else float("nan"))
        svi_gps = 0.0 if in_tunnel else (len(gps_los) / len(gps_looks) if gps_looks else float("nan"))
        svi_gal = 0.0 if in_tunnel else (len(gal_los) / len(gal_looks) if gal_looks else float("nan"))
        epoch_rows.append({
            "epoch_iso": epoch.isoformat(), "t_s": t, "dist_m": dist_m, "lon": lon, "lat": lat,
            "visible_total": visible, "los_total": los, "nlos_obstacles": nlos, "vegetation_degraded": veg,
            "gnss_denied_tunnel_or_covered": 1 if in_tunnel else 0,
            "pdop_present": pdop_present, "pdop_los_total": float("nan") if in_tunnel else pdop_los,
            "pdop_los_gps": float("nan") if in_tunnel else pdop_gps,
            "pdop_los_galileo": float("nan") if in_tunnel else pdop_gal,
            "svi_total": svi_total, "svi_gps": svi_gps, "svi_galileo": svi_gal
        })
    return pd.DataFrame(epoch_rows), pd.DataFrame(look_rows)


def _rows_for_score(epochs: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for r in epochs.to_dict("records"):
        rows.append({
            "los": r.get("los_total", 0),
            "sviTotal": r.get("svi_total", float("nan")),
            "pdopLos": r.get("pdop_los_total", float("nan")),
            "tunnel": r.get("gnss_denied_tunnel_or_covered", 0),
        })
    return rows


def _add_window_columns(epochs: pd.DataFrame, start_time_utc: str, score: float, window_kind: str) -> pd.DataFrame:
    out = epochs.copy()
    out.insert(0, "window_kind", window_kind)
    out.insert(1, "window_start_time_utc", start_time_utc)
    out.insert(2, "window_score", score)
    return out


def _mission_total_sec(city: dict[str, Any], config: dict[str, Any]) -> int:
    speed_mps = float(config["survey"].get("speed_kmh", 30.0)) / 3.6
    return max(0, int(math.floor(city["route_length_m"] / max(0.01, speed_mps))))


def predict_windows(city: dict[str, Any], sats: list[Satellite], config: dict[str, Any], progress=print) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Port of predictBestSurveyTime(): daily preview + optional full refinement.

    By default it follows the HTML logic: quick preview from 06:00 to 22:00 every
    30 min without vegetation, then refine best/worst candidates with the full
    vegetation-aware model. Config can override start/end/step for Toolbox runs.
    """
    pcfg = config.get("planning", {})
    if not pcfg.get("enabled", True):
        return pd.DataFrame(), {}
    date = config["survey"]["date_utc"]
    start_s = pcfg.get("start_time_utc", "06:00:00")
    end_s = pcfg.get("end_time_utc", "22:00:00")
    step_min = float(pcfg.get("step_minutes", 30))
    start = _dt(date, start_s)
    end = _dt(date, end_s)
    step = timedelta(minutes=step_min)
    preview_rows = []
    cur = start; k = 0
    total_sec = _mission_total_sec(city, config)
    preview_max_epochs = int(pcfg.get("preview_max_epochs", 42))
    refinement_max_epochs = int(pcfg.get("refinement_max_epochs", 90))
    include_vegetation = bool(pcfg.get("include_vegetation", True))
    if progress:
        progress(f"[AetherMMS]       Preview pass: {start_s}-{end_s} scan, maxEpochs={preview_max_epochs}, vegetation skipped.")
    while cur <= end:
        if progress:
            progress(f"[AetherMMS]       Fast temporal preview - {_format_time(cur)}...")
        epochs, _ = simulate_mission(city, sats, config, cur, quick=True, max_epochs=preview_max_epochs, skip_looks=True)
        s = score_survey_window(_rows_for_score(epochs))
        preview_rows.append({"start_time_utc": _format_time(cur), "end_time_utc": _format_time(cur + timedelta(seconds=total_sec)), "mode": "preview", **s})
        k += 1
        cur += step
    if not preview_rows:
        return pd.DataFrame(), {}

    sorted_preview_desc = sorted(preview_rows, key=lambda r: r["score"], reverse=True)
    seed_count = int(pcfg.get("seed_count", 6))
    offset_min = int(pcfg.get("refine_offset_minutes", 15))
    day_start_min = start.hour * 60 + start.minute
    day_end_min = end.hour * 60 + end.minute
    seed_minutes: set[int] = set()

    if progress:
        progress(f"[AetherMMS]       Refinement seeds: {seed_count} best + {seed_count} worst preview windows, offsets -{offset_min}/0/+{offset_min} min.")
    for cand in [*sorted_preview_desc[:seed_count], *sorted_preview_desc[-seed_count:]]:
        hh, mm, _ = [int(x) for x in cand["start_time_utc"].split(":")]
        m = hh * 60 + mm
        for x in (m - offset_min, m, m + offset_min):
            if day_start_min <= x <= day_end_min:
                seed_minutes.add(x)

    final_rows = []
    refined_tracks: dict[str, pd.DataFrame] = {}
    for idx, minutes in enumerate(sorted(seed_minutes), start=1):
        hh = str(minutes // 60).zfill(2)
        mm = str(minutes % 60).zfill(2)
        ts = f"{hh}:{mm}:00"
        if progress:
            veg_label = "enabled" if include_vegetation else "skipped"
            progress(f"[AetherMMS]       Window refinement {idx}/{len(seed_minutes)} - {ts}, maxEpochs={refinement_max_epochs}, vegetation {veg_label}...")
        epochs, _ = simulate_mission(city, sats, config, _dt(date, ts), quick=not include_vegetation, max_epochs=refinement_max_epochs, skip_looks=True)
        s = score_survey_window(_rows_for_score(epochs))
        end_ts = _format_time(_dt(date, ts) + timedelta(seconds=total_sec))
        final_rows.append({"start_time_utc": ts, "end_time_utc": end_ts, "mode": "refined", **s})
        refined_tracks[ts] = epochs

    if not final_rows:
        df = pd.DataFrame(preview_rows).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df, {}

    df = pd.DataFrame(final_rows).sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    best = df.iloc[0]
    worst = df.iloc[-1]
    tracks: dict[str, pd.DataFrame] = {}
    for kind, row in (("best", best), ("worst", worst)):
        ts = row["start_time_utc"]
        track = refined_tracks.get(ts)
        if track is None:
            continue
        tracks[kind] = _add_window_columns(track, ts, float(row["score"]), kind)
    return df, tracks
