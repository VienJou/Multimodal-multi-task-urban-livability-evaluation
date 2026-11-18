"""Interactive map of livability scores on the 100 m test grid cells.

Each test sample is a 100 m x 100 m grid cell whose name encodes its position in
the Dutch RD New grid (EPSG:28992), e.g. ``RS_E1993N3660.tif`` -> x = 199300 m,
y = 366000 m (taken as the lower-left corner of the cell). This module turns the
JSONL files in ``json/`` into cell polygons and draws them on a folium map,
optionally with the model predictions written by ``evaluate()`` in the main
notebook.

Only ``numpy``, ``pandas`` and ``folium`` are needed: no GPU, no Hugging Face
token and no model checkpoint. All paths are relative to this file, so the code
runs unchanged locally and on the I-GUIDE platform.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "json"
CELL_SIZE_M = 100

CITY_FILES: Dict[str, str] = {
    "Beesel": "Livability_test_Beesel_0320.json",
    "Dordrecht": "Livability_test_Dordrecht_0320.json",
    "Eindhoven": "Livability_test_Eindhoven_0320.json",
    "Hengelo": "Livability_test_Hengelo_0320.json",
}

# Dataset key -> label used in the paper (Dutch Leefbaarometer code in brackets)
ASPECTS: Dict[str, str] = {
    "lbm": "Overall livability (LBM)",
    "fys": "Physical environment, PHY (FYS)",
    "onv": "Nuisance and insecurity, NUI (ONV)",
    "soc": "Social cohesion, SOC (SOC)",
    "vrz": "Amenities, AME (VRZ)",
    "won": "Housing quality, HOU (WON)",
}

# Colour scales: LBM is a magnitude (sequential, one hue); the five dimensions
# are signed contributions around 0 (diverging, gray midpoint).
SEQUENTIAL_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
DIVERGING_RED_BLUE = ["#e34948", "#f0efec", "#256abf"]

GRAY_CANVAS_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
                   "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}")

_CELL_RE = re.compile(r"E(\d+)N(\d+)")


def rd_to_wgs84(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert RD New (EPSG:28992) metres to WGS84 latitude/longitude.

    Uses the polynomial approximation of Schreutelkamp & Strang van Hees,
    accurate to about 1 m inside the Netherlands - well below the 100 m cell size.

    Args:
        x: RD x coordinates in metres.
        y: RD y coordinates in metres.

    Returns:
        Tuple ``(lat, lon)`` in decimal degrees.
    """
    dx = (np.asarray(x, dtype=float) - 155000.0) * 1e-5
    dy = (np.asarray(y, dtype=float) - 463000.0) * 1e-5
    lat_terms = [
        (0, 1, 3235.65389), (2, 0, -32.58297), (0, 2, -0.24750),
        (2, 1, -0.84978), (0, 3, -0.06550), (2, 2, -0.01709),
        (1, 0, -0.00738), (4, 0, 0.00530), (2, 3, -0.00039),
        (4, 1, 0.00033), (1, 1, -0.00012),
    ]
    lon_terms = [
        (1, 0, 5260.52916), (1, 1, 105.94684), (1, 2, 2.45656),
        (3, 0, -0.81885), (1, 3, 0.05594), (3, 1, -0.05607),
        (0, 1, 0.01199), (3, 2, -0.00256), (1, 4, 0.00128),
        (0, 2, 0.00022), (2, 0, -0.00022), (5, 0, 0.00026),
    ]
    lat = 52.15517440 + sum(k * dx**p * dy**q for p, q, k in lat_terms) / 3600.0
    lon = 5.38720621 + sum(k * dx**p * dy**q for p, q, k in lon_terms) / 3600.0
    return lat, lon


def parse_cell(name: str) -> Tuple[int, int]:
    """Return the RD (x, y) corner in metres from a name like ``RS_E1993N3660.tif``."""
    match = _CELL_RE.search(name)
    if match is None:
        raise ValueError(f"Cannot parse grid cell from '{name}'")
    return int(match.group(1)) * 100, int(match.group(2)) * 100


def load_cells(cities: Optional[Iterable[str]] = None, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load test grid cells with their reference scores.

    Args:
        cities: City names from ``CITY_FILES``; ``None`` loads all four.
        data_dir: Directory holding the JSONL files (default: ``json/`` next to this file).

    Returns:
        DataFrame with columns ``city, id, rd_x, rd_y`` and one column per aspect.
    """
    cities = list(CITY_FILES) if cities is None else list(cities)
    unknown = [c for c in cities if c not in CITY_FILES]
    if unknown:
        raise ValueError(f"Unknown city {unknown}; choose from {list(CITY_FILES)}")

    rows: List[dict] = []
    for city in cities:
        path = Path(data_dir) / CITY_FILES[city]
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rd_x, rd_y = parse_cell(rec["img"])
                row = {"city": city, "id": rec["id"], "rd_x": rd_x, "rd_y": rd_y}
                row.update({k: float(rec[k]) for k in ASPECTS})
                rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Loaded %d cells from %s", len(df), cities)
    return df


def load_predictions(csv_path: Path) -> Optional[pd.DataFrame]:
    """Read ``test_predictions.csv`` written by ``evaluate()``; ``None`` if absent."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _cell_polygons(df: pd.DataFrame) -> np.ndarray:
    """Return an (n, 5, 2) array of closed [lon, lat] rings, one per cell."""
    x0, y0 = df["rd_x"].to_numpy(), df["rd_y"].to_numpy()
    s = CELL_SIZE_M
    corners = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + s), (x0, y0 + s), (x0, y0)]
    rings = []
    for cx, cy in corners:
        lat, lon = rd_to_wgs84(cx, cy)
        rings.append(np.stack([lon, lat], axis=-1))
    return np.round(np.stack(rings, axis=1), 6)


def _feature_collection(df: pd.DataFrame, props: List[str]) -> dict:
    rings = _cell_polygons(df)
    records = df[props].to_dict("records")
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
            "properties": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()},
        }
        for ring, rec in zip(rings, records)
    ]
    return {"type": "FeatureCollection", "features": features}


def _colormap(values: np.ndarray, aspect: str):
    import branca.colormap as cm

    if aspect == "lbm":
        vmin, vmax = np.nanpercentile(values, [2, 98])
        return cm.LinearColormap(SEQUENTIAL_BLUE, vmin=vmin, vmax=vmax,
                                 caption=f"{ASPECTS[aspect]} - score")
    bound = float(np.nanpercentile(np.abs(values), 98))
    return cm.LinearColormap(DIVERGING_RED_BLUE, vmin=-bound, vmax=bound,
                             caption=f"{ASPECTS[aspect]} - below (red) / above (blue) average")


def build_map(cells: pd.DataFrame, aspect: str = "lbm",
              predictions: Optional[pd.DataFrame] = None):
    """Draw reference scores (and optional predictions) as 100 m cells on a folium map.

    Args:
        cells: Output of ``load_cells``.
        aspect: One of ``ASPECTS``.
        predictions: Optional output of ``load_predictions``; cells are matched by ``id``.

    Returns:
        ``folium.Map`` with one toggleable layer per city, plus a prediction layer
        when predictions for the shown cells are available.
    """
    import folium

    if aspect not in ASPECTS:
        raise ValueError(f"Unknown aspect '{aspect}'; choose from {list(ASPECTS)}")
    if cells.empty:
        raise ValueError("No cells to draw")

    colormap = _colormap(cells[aspect].to_numpy(), aspect)
    lat_c, lon_c = rd_to_wgs84(cells["rd_x"].mean(), cells["rd_y"].mean())
    fmap = folium.Map(location=[float(lat_c), float(lon_c)], zoom_start=13, tiles=None)
    # Keyless basemaps; the last one added (OpenStreetMap) is shown first
    folium.TileLayer(tiles=GRAY_CANVAS_URL, attr="Tiles &copy; Esri", name="Light gray canvas").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(fmap)

    def style(feature, key=aspect, outline="#ffffff"):
        return {"fillColor": colormap(feature["properties"][key]), "fillOpacity": 0.85,
                "color": outline, "weight": 0.3}

    props = ["city", "id", "rd_x", "rd_y", aspect]
    for city, group in cells.groupby("city", sort=False):
        folium.GeoJson(
            _feature_collection(group, props),
            name=f"{city}: reference ({len(group)} cells)",
            style_function=style,
            highlight_function=lambda f: {"weight": 2, "color": "#1a1a19"},
            tooltip=folium.GeoJsonTooltip(fields=props,
                                          aliases=["City", "Cell id", "RD x", "RD y", aspect.upper()]),
        ).add_to(fmap)

    pred_col = f"{aspect}_pred"
    if predictions is not None and pred_col in predictions:
        merged = cells.merge(predictions[["id", pred_col]], on="id", how="inner")
        if merged.empty:
            logger.info("No predictions match the cells on this map")
        else:
            merged["error"] = merged[pred_col] - merged[aspect]
            pprops = ["city", "id", aspect, pred_col, "error"]
            folium.GeoJson(
                _feature_collection(merged, pprops),
                name=f"Model prediction ({len(merged)} cells)",
                style_function=lambda f: {**style(f, key=pred_col, outline="#1a1a19"), "weight": 1.5},
                tooltip=folium.GeoJsonTooltip(fields=pprops,
                                              aliases=["City", "Cell id", "Reference", "Predicted",
                                                       "Error (pred - ref)"]),
            ).add_to(fmap)
    elif predictions is not None:
        logger.warning("Prediction file has no column '%s'", pred_col)

    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    bounds = rd_to_wgs84(np.array([cells["rd_x"].min(), cells["rd_x"].max() + CELL_SIZE_M]),
                         np.array([cells["rd_y"].min(), cells["rd_y"].max() + CELL_SIZE_M]))
    fmap.fit_bounds([[float(bounds[0][0]), float(bounds[1][0])],
                     [float(bounds[0][1]), float(bounds[1][1])]])
    return fmap


__all__ = ["ASPECTS", "CITY_FILES", "build_map", "load_cells", "load_predictions",
           "parse_cell", "rd_to_wgs84"]
