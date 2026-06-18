"""PNG figures and HTML report corresponding to the AetherMMS panels."""
from __future__ import annotations

from pathlib import Path
import math
import html
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform


HTML = {
    "bg": "#071020",
    "panel": "#0d1326",
    "grid": "#2a4080",
    "text": "#e0e8ff",
    "muted": "#9ab7e8",
    "total": "#9dccff",
    "gps": "#4a90e2",
    "galileo": "#ff9800",
    "nlos": "#f44336",
    "veg": "#ffc107",
    "good": "#00e676",
    "medium": "#ff9800",
    "poor": "#f44336",
    "casing": "#05070d",
    "start": "#00e676",
    "end": "#ffeb3b",
    "building": "#7a8190",
    "tree": "#2fbf71",
}


def _style_axes(ax) -> None:
    ax.set_facecolor(HTML["bg"])
    ax.figure.set_facecolor(HTML["panel"])
    ax.tick_params(colors=HTML["muted"])
    for spine in ax.spines.values():
        spine.set_color(HTML["grid"])
    ax.xaxis.label.set_color(HTML["text"])
    ax.yaxis.label.set_color(HTML["text"])
    ax.title.set_color(HTML["text"])
    ax.grid(True, color=HTML["grid"], alpha=0.32)


def _style_legend(leg) -> None:
    if leg is None:
        return
    leg.get_frame().set_facecolor(HTML["panel"])
    leg.get_frame().set_edgecolor(HTML["grid"])
    for txt in leg.get_texts():
        txt.set_color(HTML["text"])


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _pdop_label(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.2f}"


def _epoch_label(epochs: pd.DataFrame, selected_t: float | None = None) -> str:
    if epochs.empty or "epoch_iso" not in epochs.columns:
        return "UTC"
    if selected_t is None:
        row = epochs.iloc[0]
    else:
        idx = (epochs["t_s"] - selected_t).abs().idxmin()
        row = epochs.loc[idx]
    raw = str(row["epoch_iso"]).replace("T", " ")
    raw = raw.replace("+00:00", " UTC").replace("Z", " UTC")
    return raw


def _quality_color(row: pd.Series) -> str:
    if int(row.get("gnss_denied_tunnel_or_covered", row.get("tunnel", 0)) or 0):
        return HTML["poor"]
    los = float(row.get("los_total", row.get("los", 0)) or 0)
    if los <= 4:
        return HTML["poor"]
    if los <= 7:
        return HTML["medium"]
    return HTML["good"]


def _plot_features(ax, city: dict | None) -> None:
    """Draw building footprints and tree crowns directly in longitude/latitude.

    This function is only for graphical output. It does not modify the
    computational reference system used by AetherMMS: GNSS look angles remain
    ENU/ECEF, and ray-tracing still uses the local metric scene prepared in
    buildingcity.py.
    """
    if not city:
        return

    building_patches = []
    for feat in city.get("buildings", {}).get("features", []):
        try:
            geom = shape(feat["geometry"])
            geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
            for poly in geoms:
                if poly.is_empty or not hasattr(poly, "exterior"):
                    continue
                arr = np.asarray(poly.exterior.coords, dtype=float)
                building_patches.append(MplPolygon(arr, closed=True))
        except Exception:
            continue
    if building_patches:
        ax.add_collection(PatchCollection(building_patches, facecolor=HTML["building"], edgecolor=HTML["building"], linewidth=0.2, alpha=0.24, zorder=1))

    tree_patches = []
    for feat in city.get("trees", {}).get("features", []):
        try:
            geom = shape(feat["geometry"])
            props = feat.get("properties") or {}
            if geom.geom_type == "Polygon":
                geoms = [geom]
            elif geom.geom_type == "MultiPolygon":
                geoms = list(geom.geoms)
            else:
                lon = float(props.get("_tree_center_lon", geom.centroid.x))
                lat = float(props.get("_tree_center_lat", geom.centroid.y))
                radius_m = max(0.25, float(props.get("_crown_radius_m", 2.0) or 2.0))
                dlat = radius_m / 111320.0
                dlon = radius_m / (111320.0 * max(1e-6, math.cos(math.radians(lat))))
                theta = np.linspace(0.0, 2.0 * math.pi, 32)
                arr = np.column_stack([lon + dlon * np.cos(theta), lat + dlat * np.sin(theta)])
                tree_patches.append(MplPolygon(arr, closed=True))
                continue
            for poly in geoms:
                if poly.is_empty or not hasattr(poly, "exterior"):
                    continue
                arr = np.asarray(poly.exterior.coords, dtype=float)
                tree_patches.append(MplPolygon(arr, closed=True))
        except Exception:
            continue
    if tree_patches:
        ax.add_collection(PatchCollection(tree_patches, facecolor=HTML["tree"], edgecolor=HTML["tree"], linewidth=0.2, alpha=0.18, zorder=1.3))

def _plot_quality_line(ax, data: pd.DataFrame, scene=None, linewidth: float = 4.0) -> None:
    """Plot trajectory in longitude/latitude with HTML LOS colours.

    The `scene` argument is deliberately ignored for plotting. It remains in the
    function signature so callers do not change, but the figure is drawn from
    the exported lon/lat columns. Numerical GNSS processing remains ENU/ECEF.
    """
    if data.empty or len(data) < 2:
        return
    pts = data[["lon", "lat"]].to_numpy(float)
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    casing = LineCollection(segments, colors=HTML["casing"], linewidths=linewidth + 3.0, capstyle="round", joinstyle="round", zorder=3)
    ax.add_collection(casing)
    colors = [_quality_color(row) for _, row in data.iloc[:-1].iterrows()]
    line = LineCollection(segments, colors=colors, linewidths=linewidth, capstyle="round", joinstyle="round", zorder=4)
    ax.add_collection(line)
    ax.scatter(pts[0, 0], pts[0, 1], s=44, color=HTML["start"], edgecolor=HTML["casing"], linewidth=1.2, zorder=5)
    ax.scatter(pts[-1, 0], pts[-1, 1], s=44, color=HTML["end"], edgecolor=HTML["casing"], linewidth=1.2, zorder=5)
    ax.autoscale()
    lat0 = float(np.nanmean(pts[:, 1]))
    ax.set_aspect(1.0 / max(1e-6, math.cos(math.radians(lat0))), adjustable="datalim")

def _quality_legend(ax, include_scene: bool = False) -> None:
    handles = [
        Line2D([0], [0], color=HTML["poor"], lw=4, label="0-4 LOS / tunnel"),
        Line2D([0], [0], color=HTML["medium"], lw=4, label="5-7 LOS"),
        Line2D([0], [0], color=HTML["good"], lw=4, label=">7 LOS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["start"], markeredgecolor=HTML["casing"], label="START"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["end"], markeredgecolor=HTML["casing"], label="END"),
    ]
    if include_scene:
        handles.extend([
            Line2D([0], [0], color=HTML["building"], lw=6, alpha=0.45, label="Buildings"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["tree"], markeredgecolor=HTML["tree"], markersize=8, alpha=0.45, label="Trees"),
        ])
    _style_legend(ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.82))


def _two_axis_plot(
    data: pd.DataFrame,
    series: list[tuple[str, str, str, float]],
    title: str,
    ylabel: str,
    out_path: Path,
    dpi: int,
    ylim: tuple[float, float] | None = None,
    context: str = "",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.4), sharey=True)
    suffix = f" - {context}" if context else ""
    axes[0].set_title(title + " vs time" + suffix)
    axes[1].set_title(title + " vs progressive distance" + suffix)
    x_defs = [
        (data["t_s"] / 60.0, "Mission time [min]"),
        (data["dist_m"], "Progressive distance [m]"),
    ]
    for ax, (x, xlabel) in zip(axes, x_defs):
        for col, label, color, width in series:
            ax.plot(x, data[col], label=label, color=color, linewidth=width)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        _style_axes(ax)
        _style_legend(ax.legend(loc="best", fontsize=8))
    _save(fig, out_path, dpi)


def plot_satellite_counts(epochs: pd.DataFrame, out: Path, dpi: int = 180) -> None:
    _two_axis_plot(
        epochs,
        [
            ("visible_total", "Present", HTML["total"], 2.6),
            ("los_total", "LOS", HTML["gps"], 2.2),
            ("nlos_obstacles", "NLOS obstacles", HTML["nlos"], 2.0),
            ("vegetation_degraded", "Vegetation-degraded", HTML["veg"], 2.0),
        ],
        "LOS / NLOS / vegetation / present satellite count",
        "Satellite count",
        out / "satellite_count_los_nlos_vegetation_present.png",
        dpi,
        context=_epoch_label(epochs),
    )


def plot_pdop(epochs: pd.DataFrame, out: Path, dpi: int = 180) -> None:
    _two_axis_plot(
        epochs,
        [
            ("pdop_los_total", "Total PDOP LOS", HTML["total"], 2.6),
            ("pdop_los_gps", "GPS PDOP LOS", HTML["gps"], 2.2),
            ("pdop_los_galileo", "Galileo PDOP LOS", HTML["galileo"], 2.2),
        ],
        "PDOP LOS",
        "PDOP",
        out / "pdop_los_profile.png",
        dpi,
        ylim=(0, max(8.0, float(np.nanmax(epochs[["pdop_los_total", "pdop_los_gps", "pdop_los_galileo"]].replace([np.inf, -np.inf], np.nan).to_numpy())) * 1.05)),
        context=_epoch_label(epochs),
    )


def plot_svi(epochs: pd.DataFrame, out: Path, dpi: int = 180) -> None:
    _two_axis_plot(
        epochs,
        [
            ("svi_total", "Total SVI", HTML["total"], 2.6),
            ("svi_gps", "GPS SVI", HTML["gps"], 2.2),
            ("svi_galileo", "Galileo SVI", HTML["galileo"], 2.2),
        ],
        "Sky Visibility Index",
        "SVI",
        out / "sky_visibility_index.png",
        dpi,
        ylim=(0, 1.05),
        context=_epoch_label(epochs),
    )


def _select_skyplot_epoch(epochs: pd.DataFrame, mode: str = "time", value: float | None = None) -> tuple[float, str]:
    if epochs.empty:
        return 0.0, "selected epoch"
    mode = (mode or "time").strip().lower()
    if value is None:
        row = epochs.iloc[len(epochs) // 2]
        return float(row["t_s"]), f"middle epoch, distance {float(row.get('dist_m', 0.0)):.3f} m"
    if mode == "distance":
        idx = (epochs["dist_m"].astype(float) - float(value)).abs().idxmin()
        row = epochs.loc[idx]
        return float(row["t_s"]), f"distance {float(row.get('dist_m', value)):.3f} m"
    idx = (epochs["t_s"].astype(float) - float(value)).abs().idxmin()
    row = epochs.loc[idx]
    return float(row["t_s"]), f"time {float(row.get('t_s', value)):.1f} s"


def plot_skyplot(looks: pd.DataFrame, epochs: pd.DataFrame, out: Path, mode: str = "time", value: float | None = None, dpi: int = 180) -> None:
    selected_t, selector_label = _select_skyplot_epoch(epochs, mode, value)
    sl = looks[np.abs(looks["t_s"] - selected_t) <= 0.5]
    fig = plt.figure(figsize=(7.4, 6.2))
    fig.set_facecolor(HTML["panel"])
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor(HTML["bg"])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(90, 0)
    ax.set_yticks([0, 30, 60, 90])
    ax.set_yticklabels([])
    ax.tick_params(colors=HTML["muted"])
    ax.grid(True, color=HTML["grid"], alpha=0.4)
    status_colors = {"LOS": HTML["gps"], "NLOS": HTML["nlos"], "VEG": HTML["veg"], "GNSS_DENIED": HTML["nlos"]}
    status_edges = {"LOS": "#d6e8ff", "NLOS": "#ffccd0", "VEG": "#fff3c4", "GNSS_DENIED": "#ffccd0"}
    for _, row in sl.iterrows():
        theta = math.radians(float(row["az"]))
        r = 90.0 - float(row["el"])
        constellation = str(row.get("constellation", ""))
        marker = "*" if constellation == "GPS" or str(row["sv"]).upper().startswith("G") else "o"
        color = status_colors.get(str(row["status"]), HTML["total"])
        edge = status_edges.get(str(row["status"]), HTML["text"])
        ax.scatter([theta], [r], s=86 if marker == "*" else 58, marker=marker, color=color, edgecolor=edge, linewidth=0.8, zorder=3)
        ax.text(theta, r, str(row["sv"]), fontsize=7, color=HTML["text"], zorder=4)
    ax.set_title(f"Skyplot at {selector_label} - {_epoch_label(epochs, selected_t)}", color=HTML["text"])
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=HTML["gps"], markeredgecolor="#d6e8ff", markersize=12, label="GPS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["gps"], markeredgecolor="#d6e8ff", markersize=8, label="Galileo"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["nlos"], markeredgecolor="#ffccd0", markersize=8, label="NLOS / denied"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=HTML["veg"], markeredgecolor="#fff3c4", markersize=8, label="Vegetation"),
    ]
    _style_legend(ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.18), ncol=2, fontsize=8))
    _save(fig, out / "skyplot_selected_epoch.png", dpi)


def plot_trajectory_quality(epochs: pd.DataFrame, out: Path, dpi: int = 180, city: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    _plot_features(ax, city)
    _plot_quality_line(ax, epochs, scene=None, linewidth=5.0)
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Trajectory quality by LOS satellite count - {_epoch_label(epochs)}")
    _style_axes(ax)
    _quality_legend(ax, include_scene=city is not None)
    _save(fig, out / "trajectory_los_quality.png", dpi)


def _continuity_pct(track: pd.DataFrame) -> float:
    scoring = track[track["gnss_denied_tunnel_or_covered"].fillna(0).astype(float) <= 0]
    if scoring.empty:
        scoring = track
    pdop = scoring["pdop_los_total"].replace([np.inf, -np.inf], np.nan)
    outage = (scoring["los_total"].fillna(0).astype(float) < 4) | pdop.isna() | (pdop > 8)
    return 100.0 * (1.0 - float(outage.mean()))


def _summary_for_track(track: pd.DataFrame) -> str:
    score = float(track.get("window_score", pd.Series([float("nan")])).iloc[0])
    avg_los = float(track["los_total"].mean())
    pdops = track["pdop_los_total"].replace([np.inf, -np.inf], np.nan).dropna()
    avg_pdop = float(pdops.mean()) if len(pdops) else float("nan")
    continuity = _continuity_pct(track)
    return f"Score {score:.1f}/100 | LOS {avg_los:.1f} | PDOP {_pdop_label(avg_pdop)} | continuity {continuity:.0f}%"


def plot_temporal_score_line(windows: pd.DataFrame, out: Path, dpi: int = 180) -> None:
    if windows.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered = windows.sort_values("start_time_utc")
    ax.plot(ordered["start_time_utc"], ordered["score"], marker="o", color=HTML["total"], linewidth=2.4)
    ordered = windows.sort_values("score", ascending=False, kind="mergesort")
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    ax.scatter([best["start_time_utc"]], [best["score"]], color=HTML["good"], s=80, zorder=4, label="Best")
    ax.scatter([worst["start_time_utc"]], [worst["score"]], color=HTML["poor"], s=80, zorder=4, label="Worst")
    ax.set_xlabel("Candidate start time UTC")
    ax.set_ylabel("Mission score")
    ax.set_title("Refined temporal-window scores")
    ax.tick_params(axis="x", rotation=60)
    _style_axes(ax)
    _style_legend(ax.legend())
    _save(fig, out / "temporal_window_scores.png", dpi)


def plot_best_worst(windows: pd.DataFrame, csv_dir: Path, out: Path, dpi: int = 180, city: dict | None = None) -> None:
    best_path = csv_dir / "best_window_trajectory.csv"
    worst_path = csv_dir / "worst_window_trajectory.csv"
    if not best_path.exists() or not worst_path.exists():
        plot_temporal_score_line(windows, out, dpi)
        return
    best = pd.read_csv(best_path)
    worst = pd.read_csv(worst_path)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), sharex=False, sharey=False)
    for ax, data, label in [(axes[0], best, "Best time window"), (axes[1], worst, "Worst time window")]:
        _plot_features(ax, city)
        _plot_quality_line(ax, data, scene=None, linewidth=5.0)
        start = data.get("window_start_time_utc", pd.Series(["-"])).iloc[0]
        ax.set_title(f"{label} - {start}")
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        _style_axes(ax)
        ax.text(0.02, 0.02, _summary_for_track(data), transform=ax.transAxes, color=HTML["text"], fontsize=8, va="bottom", ha="left", bbox={"facecolor": HTML["panel"], "edgecolor": HTML["grid"], "alpha": 0.82})
    _quality_legend(axes[0], include_scene=city is not None)
    fig.suptitle("Best / worst temporal windows - trajectory comparison", color=HTML["text"])
    _save(fig, out / "best_worst_temporal_windows.png", dpi)


def _read_metadata(metadata_path: Path | None) -> dict[str, str]:
    if not metadata_path or not metadata_path.exists():
        return {}
    out: dict[str, str] = {}
    current = ""
    for raw in metadata_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line or line.startswith("=") or line.startswith("AetherMMS"):
            continue
        if line.startswith("  ") and current and ":" in line:
            k, v = line.strip().split(":", 1)
            out[f"{current}.{k.strip()}"] = v.strip()
        elif ":" in line:
            k, v = line.split(":", 1)
            current = k.strip()
            if v.strip():
                out[current] = v.strip()
    return out


def _fmt_float(value: object, digits: int = 1, suffix: str = "") -> str:
    try:
        v = float(value)
        if not math.isfinite(v):
            return "-"
        return f"{v:.{digits}f}{suffix}"
    except Exception:
        return "-"


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _figure_card(fig_dir: Path, report_dir: Path, name: str, title: str, caption: str) -> str:
    path = fig_dir / name
    if not path.exists():
        return ""
    src = html.escape(_rel(path, report_dir))
    return f"""
      <article class="figure-card">
        <div class="figure-copy">
          <h3>{html.escape(title)}</h3>
          <p>{html.escape(caption)}</p>
        </div>
        <a href="{src}" target="_blank" rel="noopener">
          <img src="{src}" alt="{html.escape(title)}">
        </a>
      </article>
    """


def _make_html_report(fig_dir: Path, report_path: Path, metadata_path: Path | None = None, csv_dir: Path | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = report_path.parent
    meta = _read_metadata(metadata_path)

    windows = pd.DataFrame()
    if csv_dir and (csv_dir / "best_window_summary.csv").exists():
        windows = pd.read_csv(csv_dir / "best_window_summary.csv")
    ordered_windows = windows.sort_values("score", ascending=False, kind="mergesort") if not windows.empty else windows
    best = ordered_windows.iloc[0] if not ordered_windows.empty else None
    worst = ordered_windows.iloc[-1] if not ordered_windows.empty else None

    logo_html = ""

    # Report card only: metre resolution is enough for display; CSV/metadata keep 0.001 m.
    route_m = _fmt_float(meta.get("route_length_m", "-"), 0, " m")
    speed = _fmt_float(meta.get("survey_speed_kmh", "-"), 1, " km/h")
    survey_dt = f"{meta.get('survey_date_utc', '-')} {meta.get('survey_start_time_utc', '-')}"
    base_status = html.escape(meta.get("base_coverage.status", "none")).lower()
    base_distance = _fmt_float(meta.get("base_coverage.max_km", "-"), 2, " km")

    best_time = html.escape(str(best["start_time_utc"])) if best is not None else "-"
    worst_time = html.escape(str(worst["start_time_utc"])) if worst is not None else "-"
    best_score = _fmt_float(best["score"], 1, "/100") if best is not None else "-"
    worst_score = _fmt_float(worst["score"], 1, "/100") if worst is not None else "-"
    best_los = _fmt_float(best["avg_los"], 1) if best is not None else "-"
    worst_los = _fmt_float(worst["avg_los"], 1) if worst is not None else "-"
    best_pdop = _fmt_float(best["avg_pdop"], 2) if best is not None else "-"
    worst_pdop = _fmt_float(worst["avg_pdop"], 2) if worst is not None else "-"
    best_cont = _fmt_float(float(best["continuity_score"]) * 100.0, 0, "%") if best is not None else "-"
    worst_cont = _fmt_float(float(worst["continuity_score"]) * 100.0, 0, "%") if worst is not None else "-"

    figures = [
        ("best_worst_temporal_windows.png", "Best / Worst Temporal Windows", "Side-by-side trajectory comparison, with LOS class coloring and transparent buildings/trees."),
        ("temporal_window_scores.png", "Temporal Scores", "Refined candidate score distribution used to select final best and worst windows."),
        ("trajectory_los_quality.png", "Trajectory LOS Quality", "Full mission trajectory colored by LOS satellite count with planimetric urban context."),
        ("satellite_count_los_nlos_vegetation_present.png", "Satellite Counts", "Present, LOS, NLOS and vegetation-degraded satellites vs time and progressive distance."),
        ("pdop_los_profile.png", "PDOP LOS", "PDOP computed from pure LOS satellites, shown by time and progressive distance."),
        ("sky_visibility_index.png", "Sky Visibility Index", "SVI for Total, GPS and Galileo, shown by time and progressive distance."),
        ("skyplot_selected_epoch.png", "Skyplot", "Selected epoch skyplot with GPS/Galileo marker distinction and status coloring."),
    ]
    figure_html = "\n".join(_figure_card(fig_dir, report_dir, *item) for item in figures)

    css = """
    :root {
      --bg: #071020;
      --panel: #0d1326;
      --panel2: #101a33;
      --line: #274276;
      --text: #e8f0ff;
      --muted: #9ab7e8;
      --blue: #4a90e2;
      --cyan: #9dccff;
      --green: #00e676;
      --orange: #ff9800;
      --red: #f44336;
      --yellow: #ffc107;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, rgba(74,144,226,.22), transparent 34rem), var(--bg);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }
    .wrap { width: min(1180px, calc(100% - 40px)); margin: 0 auto; }
    header {
      padding: 34px 0 24px;
      border-bottom: 1px solid rgba(157,204,255,.18);
      background: linear-gradient(180deg, rgba(7,16,32,.98), rgba(7,16,32,.72));
    }
    .hero { display: block; }
    .eyebrow { color: var(--cyan); letter-spacing: 3px; text-transform: uppercase; font-size: 12px; font-weight: 700; }
    h1 { margin: 8px 0 8px; font-size: clamp(34px, 5vw, 62px); line-height: 1; letter-spacing: .08em; }
    .subtitle { color: var(--muted); max-width: 820px; margin: 0; font-size: 17px; }
    .meta-row { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 10px; }
    .chip { border: 1px solid var(--line); border-radius: 999px; padding: 7px 11px; color: var(--muted); background: rgba(13,19,38,.72); font-size: 13px; }
    .section { padding: 28px 0; }
    .section-title { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }
    h2 { margin: 0; font-size: 22px; letter-spacing: .06em; text-transform: uppercase; color: var(--cyan); }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
    .card, .window-card, .figure-card {
      border: 1px solid rgba(157,204,255,.18);
      background: linear-gradient(180deg, rgba(16,26,51,.92), rgba(9,14,28,.92));
      box-shadow: 0 12px 34px rgba(0,0,0,.22);
    }
    .card { padding: 16px; border-radius: 8px; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; }
    .value { margin-top: 8px; font-size: 26px; font-weight: 750; }
    .base-status { color: var(--green); }
    .base-status.yellow { color: var(--yellow); }
    .base-status.orange { color: var(--orange); }
    .base-status.red { color: var(--red); }
    .windows { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .window-card { padding: 18px; border-radius: 8px; }
    .window-card.best { border-color: rgba(0,230,118,.42); }
    .window-card.worst { border-color: rgba(244,67,54,.42); }
    .window-title { display:flex; justify-content:space-between; gap:12px; color: var(--muted); text-transform: uppercase; letter-spacing:.1em; font-size: 12px; }
    .time { color: var(--text); font-size: 26px; font-weight: 750; margin: 8px 0 12px; }
    .metrics { display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .metric { border-top: 1px solid rgba(157,204,255,.15); padding-top: 8px; }
    .metric strong { display:block; font-size: 18px; }
    .metric span { color: var(--muted); font-size: 12px; }
    .figure-grid { display: grid; gap: 18px; }
    .figure-card { border-radius: 8px; overflow: hidden; }
    .figure-copy { padding: 15px 16px 0; }
    .figure-copy h3 { margin: 0 0 4px; font-size: 18px; color: var(--text); }
    .figure-copy p { margin: 0 0 12px; color: var(--muted); }
    .figure-card img { display:block; width:100%; height:auto; border-top: 1px solid rgba(157,204,255,.15); background: var(--bg); }
    pre {
      margin: 0;
      padding: 16px;
      overflow:auto;
      border-radius: 8px;
      border: 1px solid rgba(157,204,255,.18);
      background: rgba(5,8,18,.7);
      color: var(--muted);
      font-size: 12px;
    }
    footer { padding: 28px 0 42px; color: var(--muted); border-top: 1px solid rgba(157,204,255,.15); }
    @media (max-width: 860px) {
      .cards { grid-template-columns: repeat(2, 1fr); }
      .windows { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
    }
    """
    metadata_text = html.escape(metadata_path.read_text(encoding="utf-8", errors="replace")) if metadata_path and metadata_path.exists() else ""
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AetherMMS Report</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <div class="wrap hero">
      {logo_html}
      <div>
        <div class="eyebrow">GNSS Planning Insight</div>
        <h1>AetherMMS</h1>
        <div class="meta-row">
          <span class="chip">Survey UTC: {html.escape(survey_dt)}</span>
          <span class="chip">Report: HTML + PNG</span>
        </div>
      </div>
    </div>
  </header>

  <main class="wrap">
    <section class="section">
      <div class="section-title"><h2>Mission Overview</h2></div>
      <div class="cards">
        <div class="card"><div class="label">Route Length</div><div class="value">{route_m}</div></div>
        <div class="card"><div class="label">Speed</div><div class="value">{speed}</div></div>
        <div class="card"><div class="label">Epochs</div><div class="value">{html.escape(meta.get("epochs", "-"))}</div></div>
        <div class="card"><div class="label">Sat.-Epoch Obs.</div><div class="value">{html.escape(meta.get("satellite_looks", "-"))}</div></div>
        <div class="card"><div class="label">Buildings</div><div class="value">{html.escape(meta.get("buildings_in_buffer", "-"))}</div></div>
        <div class="card"><div class="label">Trees</div><div class="value">{html.escape(meta.get("trees_in_buffer", "-"))}</div></div>
        <div class="card"><div class="label">Survey Bases</div><div class="value">{html.escape(meta.get("survey_bases", "-"))}</div></div>
        <div class="card"><div class="label">Base Distance</div><div class="value base-status {base_status}">{base_distance}</div></div>
        <div class="card"><div class="label">Tunnels / Covered</div><div class="value">{html.escape(meta.get("tunnels_detected", "-"))}</div></div>
      </div>
    </section>

    <section class="section">
      <div class="section-title"><h2>Temporal Prediction</h2></div>
      <div class="windows">
        <article class="window-card best">
          <div class="window-title"><span>Best Window</span><span>{best_score}</span></div>
          <div class="time">{best_time}</div>
          <div class="metrics">
            <div class="metric"><strong>{best_los}</strong><span>avg LOS</span></div>
            <div class="metric"><strong>{best_pdop}</strong><span>avg PDOP</span></div>
            <div class="metric"><strong>{best_cont}</strong><span>continuity</span></div>
            <div class="metric"><strong>{best_score}</strong><span>score</span></div>
          </div>
        </article>
        <article class="window-card worst">
          <div class="window-title"><span>Worst Window</span><span>{worst_score}</span></div>
          <div class="time">{worst_time}</div>
          <div class="metrics">
            <div class="metric"><strong>{worst_los}</strong><span>avg LOS</span></div>
            <div class="metric"><strong>{worst_pdop}</strong><span>avg PDOP</span></div>
            <div class="metric"><strong>{worst_cont}</strong><span>continuity</span></div>
            <div class="metric"><strong>{worst_score}</strong><span>score</span></div>
          </div>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="section-title"><h2>Figures</h2><span class="chip">Click any figure to open the PNG</span></div>
      <div class="figure-grid">
        {figure_html}
      </div>
    </section>

    <section class="section">
      <div class="section-title"><h2>Metadata</h2></div>
      <pre>{metadata_text}</pre>
    </section>
  </main>

  <footer>
    <div class="wrap">AetherMMS Python Core - GNSS Planning Insight</div>
  </footer>
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")


def generate_all_figures(csv_dir: Path, fig_dir: Path, report_path: Path, skyplot_mode: str = "time", skyplot_value=None, dpi: int = 180, city: dict | None = None) -> None:
    epochs = pd.read_csv(csv_dir / "epochs.csv")
    looks = pd.read_csv(csv_dir / "satellite_visibility.csv")
    plot_satellite_counts(epochs, fig_dir, dpi)
    plot_pdop(epochs, fig_dir, dpi)
    plot_svi(epochs, fig_dir, dpi)
    plot_skyplot(looks, epochs, fig_dir, skyplot_mode, skyplot_value, dpi)
    plot_trajectory_quality(epochs, fig_dir, dpi, city)
    wpath = csv_dir / "best_window_summary.csv"
    if wpath.exists():
        windows = pd.read_csv(wpath)
        plot_temporal_score_line(windows, fig_dir, dpi)
        plot_best_worst(windows, csv_dir, fig_dir, dpi, city)
    _make_html_report(fig_dir, report_path, csv_dir / "metadata.txt", csv_dir)
