"""File discovery and small I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the YAML configuration file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ignored(path: Path, root: Path, ignore_dirs: set[str]) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part in ignore_dirs for part in rel_parts)


def _is_topography(path: Path, keywords: list[str]) -> bool:
    stem = path.stem.lower()
    return any(word.lower() in stem for word in keywords)


def _score_dem(path: Path) -> tuple[int, str]:
    stem = path.stem.lower()
    score = 0
    for token in ("dem", "bare", "earth", "dtm", "lidar"):
        if token in stem:
            score -= 1
    return score, path.name.lower()


def discover_files(config: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    """Auto-detect DEM, line topography, and travel-time files."""
    root = Path(root).resolve()
    data_dir = (root / config.get("data_dir", config.get("paths", {}).get("data_dir", "."))).resolve()
    discovery = config.get("discovery", {})
    ignore_dirs = set(discovery.get("ignore_dirs", []))
    topo_keywords = discovery.get("topography_keywords", ["topo"])
    seismic_exts = {e.lower() for e in discovery.get("seismic_extensions", [".dat", ".txt"])}
    excluded_line_prefixes = tuple(
        str(prefix).lower()
        for prefix in discovery.get("exclude_line_prefixes", [])
    )
    dem_exts = {".tif", ".tiff"}

    files = [
        p
        for p in data_dir.rglob("*")
        if p.is_file() and not _ignored(p, data_dir, ignore_dirs)
    ]
    dem_files = sorted([p for p in files if p.suffix.lower() in dem_exts], key=_score_dem)
    configured_dem = config.get("dem_file")
    if configured_dem:
        configured_path = Path(str(configured_dem))
        candidates = [root / configured_path, data_dir / configured_path]
        dem_path = next((path.resolve() for path in candidates if path.exists()), None)
        if dem_path is not None:
            dem_files = [dem_path] + [p for p in dem_files if p.resolve() != dem_path]
    topo_files = sorted(
        [
            p
            for p in files
            if p.suffix.lower() in seismic_exts
            and _is_topography(p, topo_keywords)
            and not p.stem.lower().startswith(excluded_line_prefixes)
        ]
    )
    seismic_files = sorted(
        [
            p
            for p in files
            if p.suffix.lower() in seismic_exts and not _is_topography(p, topo_keywords)
            and not p.stem.lower().startswith(excluded_line_prefixes)
        ]
    )

    topo_by_stem = {p.stem.lower(): p for p in topo_files}
    lines: list[dict[str, Any]] = []
    for seismic in seismic_files:
        stem = seismic.stem
        candidates = [
            f"{stem}_topo".lower(),
            f"{stem}_topography".lower(),
            stem.lower().replace("_l1", "_l1_topo"),
            stem.lower().replace("_l2", "_l2_topo"),
        ]
        topo = next((topo_by_stem[c] for c in candidates if c in topo_by_stem), None)
        lines.append({"name": stem, "seismic": seismic, "topography": topo})
    matched_topography_files = sorted(
        {
            line["topography"]
            for line in lines
            if line["topography"] is not None
        }
    )

    return {
        "data_dir": data_dir,
        "dem_files": dem_files,
        "dem": dem_files[0] if dem_files else None,
        "topography_files": matched_topography_files,
        "seismic_files": seismic_files,
        "lines": lines,
    }


def load_numeric_table(path: str | Path) -> pd.DataFrame:
    """Read a whitespace or comma separated numeric table with comments ignored."""
    path = Path(path)
    try:
        df = pd.read_csv(path, sep=r"\s+|,", engine="python", comment="#", header=None)
    except Exception:
        df = pd.read_csv(path, delim_whitespace=True, comment="#", header=None)
    return df.dropna(axis=1, how="all")


