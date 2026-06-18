#!/usr/bin/env python3
"""Run AetherMMS Python Core from text configuration file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from buildingcity import build_city
from satellitepropagation import load_satellites
from bestwindowsprediction import predict_windows, simulate_mission
from plotting import generate_all_figures


def _parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_float(value: str, default: float) -> float:
    return default if value.strip() == "" else float(value)


def _parse_int(value: str, default: int) -> int:
    return default if value.strip() == "" else int(float(value))


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, str] = {}
    # utf-8-sig transparently accepts files saved with a UTF-8 BOM (common on Windows editors).
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or (stripped.startswith("[") and stripped.endswith("]")):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid config line {line_no}: expected key = value")
        key, value = stripped.split("=", 1)
        raw[key.strip()] = value.split("#", 1)[0].strip()

    constellations = {c.lower(): True for c in _parse_list(raw.get("constellations", "GPS, Galileo"))}
    return {
        "data_dir": raw.get("data_dir", "Data"),
        "output_dir": raw.get("output_dir", "results"),
        "inputs": {
            "dem_geotiff": raw.get("dem_geotiff", "DTM5x5.tif"),
            "trajectory_kml": raw.get("trajectory_kml", "Traj.kml"),
            "buildings_geojson": raw.get("buildings_geojson", "Buildings.geojson"),
            "trees_geojson": raw.get("trees_geojson", ""),
            "bases_kml": raw.get("bases_kml", ""),
            "gps_yuma": raw.get("gps_yuma", ""),
            "galileo_xml": raw.get("galileo_xml", ""),
        },
        "mms": {
            "lidar_range_m": _parse_float(raw.get("lidar_range_m", "100.0"), 100.0),
            "antenna_height_m": _parse_float(raw.get("antenna_height_m", "2.2"), 2.2),
        },
        "survey": {
            "date_utc": raw.get("date_utc", "2026-05-09"),
            "start_time_utc": raw.get("start_time_utc", "10:00:00"),
            "speed_kmh": _parse_float(raw.get("speed_kmh", "30.0"), 30.0),
            "elevation_mask_deg": _parse_float(raw.get("elevation_mask_deg", "10.0"), 10.0),
            "step_sec": _parse_int(raw.get("step_sec", "1"), 1),
            "constellations": {
                "gps": constellations.get("gps", False),
                "galileo": constellations.get("galileo", False),
            },
        },
        "osm": {
            "enabled": _parse_bool(raw.get("osm_enabled", "true"), True),
            "use_cache": _parse_bool(raw.get("osm_use_cache", "true"), True),
            "cache_file": raw.get("osm_cache_file", "osm_tunnels_cache.geojson"),
            "overpass_timeout_s": _parse_int(raw.get("overpass_timeout_s", "25"), 25),
            "endpoints": _parse_list(raw.get("overpass_endpoints", "")),
        },
        "planning": {
            "enabled": _parse_bool(raw.get("planning_enabled", "true"), True),
            "start_time_utc": raw.get("planning_start_time_utc", "06:00:00"),
            "end_time_utc": raw.get("planning_end_time_utc", "22:00:00"),
            "step_minutes": _parse_float(raw.get("planning_step_minutes", "30"), 30.0),
            "seed_count": _parse_int(raw.get("planning_seed_count", "6"), 6),
            "refine_offset_minutes": _parse_int(raw.get("planning_refine_offset_minutes", "15"), 15),
            "preview_max_epochs": _parse_int(raw.get("planning_preview_max_epochs", "42"), 42),
            "refinement_max_epochs": _parse_int(raw.get("planning_refinement_max_epochs", "90"), 90),
            "include_vegetation": _parse_bool(raw.get("planning_include_vegetation", "true"), True),
        },
        "ray_tracing": {
            "max_ray_km": _parse_float(raw.get("max_ray_km", "1.2"), 1.2),
        },
        "figures": {
            "dpi": _parse_int(raw.get("figure_dpi", "180"), 180),
            "skyplot_mode": raw.get("selected_skyplot_mode", "time").strip().lower() or "time",
            "skyplot_value": None if raw.get("selected_skyplot_value", raw.get("selected_skyplot_epoch_s", "")) == "" else _parse_float(raw.get("selected_skyplot_value", raw.get("selected_skyplot_epoch_s", "")), 0.0),
        },
    }


def _project_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.name.lower() == "config":
        return parent.parent
    if parent.name.lower() == "source":
        return parent.parent
    return parent


def _resolve_output(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def _metadata_txt(metadata: dict[str, Any]) -> str:
    lines = ["AetherMMS Python Core metadata", "============================", ""]
    for key, value in metadata.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _round_metre_values(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k.lower().endswith(("_m", "_metres", "_meters")):
                try:
                    out[k] = round(float(v), 3)
                except Exception:
                    out[k] = v
            else:
                out[k] = _round_metre_values(v)
        return out
    return value


def _round_metre_columns(df):
    for col in df.columns:
        name = str(col).lower()
        if name.endswith("_m") or name in {"dist_m", "range_m"}:
            try:
                df[col] = df[col].astype(float).round(3)
            except Exception:
                pass
    return df


def main() -> int:
    project_root = THIS_DIR.parent if THIS_DIR.name.lower() == "source" else THIS_DIR
    default_config = project_root / "config" / "config.txt"
    ap = argparse.ArgumentParser(description="AetherMMS Python Core runner")
    ap.add_argument("--config", default=str(default_config), help="Path to config/config.txt")
    ap.add_argument("--no-figures", action="store_true", help="Generate CSV only")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists() and not cfg_path.is_absolute():
        candidate = project_root / cfg_path
        if candidate.exists():
            cfg_path = candidate
    root = _project_root(cfg_path)
    cfg = load_config(cfg_path)
    out_dir = _resolve_output(root, cfg.get("output_dir", "results"))
    csv_dir = out_dir / "csv"
    fig_dir = out_dir / "png"
    report_path = out_dir / "AetherMMS_report.html"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[AetherMMS] 01/06 Building city: DTM, trajectory, buildings, trees, tunnels...")
    city = build_city(cfg, root)
    print(f"[AetherMMS]       Route length: {city['route_length_m']:.3f} m")
    print(f"[AetherMMS]       Buildings: {len(city['buildings']['features'])}")
    print(f"[AetherMMS]       Trees: {len(city['trees']['features'])}")
    print(f"[AetherMMS]       Tunnels/covered/overpasses: {len(city['tunnels']['features'])}")
    print(f"[AetherMMS]       Survey bases: {len(city['bases'])}")
    print(f"[AetherMMS]       Base Check Coverage: {city['base_coverage']['label']}")

    print("[AetherMMS] 02/06 Loading and propagating GNSS almanacs...")
    example_dir = city["example_dir"]
    sats = load_satellites(example_dir, cfg["inputs"], cfg["survey"].get("constellations", {}))
    gps_count = sum(1 for s in sats if s.constellation == "GPS")
    gal_count = sum(1 for s in sats if s.constellation == "Galileo")
    print(f"[AetherMMS]       Loaded satellites: {len(sats)} total ({gps_count} GPS, {gal_count} Galileo)")

    print("[AetherMMS] 03/06 Running 1 Hz GNSS visibility / 3D obstruction analysis...")
    epochs, looks = simulate_mission(city, sats, cfg)
    epochs = _round_metre_columns(epochs)
    looks = _round_metre_columns(looks)
    epochs.to_csv(csv_dir / "epochs.csv", index=False)
    looks.to_csv(csv_dir / "satellite_visibility.csv", index=False)
    epochs[["epoch_iso", "t_s", "dist_m", "pdop_present", "pdop_los_total", "pdop_los_gps", "pdop_los_galileo"]].to_csv(csv_dir / "pdop_timeseries.csv", index=False)
    print(f"[AetherMMS]       Epochs: {len(epochs)}; satellite looks: {len(looks)}")

    print("[AetherMMS] 04/06 Best/worst temporal-window prediction...")
    windows, window_tracks = predict_windows(city, sats, cfg, progress=print)
    if not windows.empty:
        windows = _round_metre_columns(windows)
        windows.to_csv(csv_dir / "best_window_summary.csv", index=False)
        for name, track in window_tracks.items():
            track = _round_metre_columns(track)
            track.to_csv(csv_dir / f"{name}_window_trajectory.csv", index=False)
        best = windows.iloc[0]
        worst = windows.iloc[-1]
        print(f"[AetherMMS]       Best:  {best['start_time_utc']} score {best['score']:.1f}/100")
        print(f"[AetherMMS]       Worst: {worst['start_time_utc']} score {worst['score']:.1f}/100")
    else:
        print("[AetherMMS]       Planning disabled or no temporal candidates generated.")

    print("[AetherMMS] 05/06 Writing metadata...")
    metadata = {
        "software": "AetherMMS Python Core",
        "survey_date_utc": cfg["survey"]["date_utc"],
        "survey_start_time_utc": cfg["survey"]["start_time_utc"],
        "survey_speed_kmh": cfg["survey"]["speed_kmh"],
        "route_length_m": round(float(city["route_length_m"]), 3),
        "epochs": int(len(epochs)),
        "satellite_looks": int(len(looks)),
        "buildings_in_buffer": len(city["buildings"]["features"]),
        "trees_in_buffer": len(city["trees"]["features"]),
        "tunnels_detected": len(city["tunnels"]["features"]),
        "survey_bases": len(city["bases"]),
        "base_coverage": _round_metre_values(city["base_coverage"]),
        "dem_stats": _round_metre_values(city["dem_stats"]),
        "html_faithful_core": True,
    }
    (csv_dir / "metadata.txt").write_text(_metadata_txt(metadata), encoding="utf-8")
    print(f"[AetherMMS]       CSV outputs written to {csv_dir}")

    if not args.no_figures:
        print("[AetherMMS] 06/06 Generating PNG figures and HTML report...")
        figs = cfg.get("figures", {})
        generate_all_figures(csv_dir, fig_dir, report_path, figs.get("skyplot_mode", "time"), figs.get("skyplot_value"), int(figs.get("dpi", 180)), city)
        print(f"[AetherMMS]       PNG figures written to {fig_dir}")
        print(f"[AetherMMS]       HTML report written to {report_path}")

    try:
        city["dem"].close()
    except Exception:
        pass
    print("[AetherMMS] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
