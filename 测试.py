from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import queue
import re
import threading
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle, Rectangle
from mlxtend.frequent_patterns import association_rules, fpgrowth

# ==========================================
# 1. Settings
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RULES_CSV_NAME = "ship_frequent_itemsets_fpgrowth.csv"
RESULTS_XLSX_NAME = "slotting_adjustments.xlsx"
BEFORE_LAYOUT_XLSX_NAME = "调仓前PD3物料关联组.xlsx"
AFTER_LAYOUT_XLSX_NAME = "调仓后PD3物料关联组.xlsx"
BEFORE_LAYOUT_SVG_NAME = "调仓前PD3物料关联组.svg"
AFTER_LAYOUT_SVG_NAME = "调仓后PD3物料关联组.svg"

ALLOW_CROSS_AISLE = True
LOCK_PASSIVE_SWAP = True
MAX_ADJUSTMENTS = None

ENV_SHIP_DATA_PATH = "RESLOTTING_SHIP_DATA_PATH"
ENV_LX03_PATH = "RESLOTTING_LX03_PATH"
ENV_BIN_SETTINGS_PATH = "RESLOTTING_BIN_SETTINGS_PATH"

def _as_path(value: Any) -> Path | None:
    if value is None: return None
    text = str(value).strip()
    if not text: return None
    path = Path(text).expanduser()
    if not path.is_absolute(): path = BASE_DIR / path
    return path.resolve()

def _config_source(config: dict[str, Any] | None, key: str) -> Path | None:
    if not config: return None
    return _as_path(config.get("data_sources", {}).get(key, ""))

def resolve_ship_data_path(config=None, override_path=None):
    override = _as_path(override_path)
    if override is not None: return override
    env = _as_path(os.getenv(ENV_SHIP_DATA_PATH, ""))
    if env is not None: return env
    return _config_source(config, "ship_data_path")

def resolve_lx03_path(config=None, override_path=None):
    override = _as_path(override_path)
    if override is not None: return override
    env = _as_path(os.getenv(ENV_LX03_PATH, ""))
    if env is not None: return env
    return _config_source(config, "lx03_path")

def resolve_bin_settings_path(config=None, override_path=None):
    override = _as_path(override_path)
    if override is not None: return override
    env = _as_path(os.getenv(ENV_BIN_SETTINGS_PATH, ""))
    if env is not None: return env
    return _config_source(config, "bin_settings_path")

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

# ==========================================
# 2. Config Manager
# ==========================================
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "whn": "M11",
    "plant": "M001",
    "start_date": "2025-10-01",
    "end_date": "2026-03-14",
    "min_support": 0.001,
    "data_sources": {
        "ship_data_path": "",
        "lx03_path": "",
        "bin_settings_path": "",
    },
    "business_rules": {
        "aisle_priority": ["07", "06"],
        "aisle_rules": [],
        "reserved_bin_patterns": [],
    },
}

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = _deep_merge(DEFAULT_CONFIG, {key: value for key, value in config.items() if key in DEFAULT_CONFIG})
    for k in ["whn", "plant", "start_date", "end_date"]:
        cfg[k] = str(cfg.get(k, "")).strip()
    cfg["min_support"] = float(cfg.get("min_support", DEFAULT_CONFIG["min_support"]))
    if not 0 < cfg["min_support"] <= 1:
        raise ValueError("min_support 必须在 (0, 1] 范围内。")
    
    sources = cfg.setdefault("data_sources", {})
    for key in ("ship_data_path", "lx03_path", "bin_settings_path"):
        sources[key] = str(sources.get(key, "") or "").strip()
        
    br = cfg.setdefault("business_rules", {})
    br["aisle_priority"] = [str(x).strip() for x in br.get("aisle_priority", []) if str(x).strip()]
    br["reserved_bin_patterns"] = [str(x).strip() for x in br.get("reserved_bin_patterns", []) if str(x).strip()]
    
    cleaned_rules: list[dict[str, Any]] = []
    for row in br.get("aisle_rules", []):
        if not isinstance(row, dict): continue
        pattern = str(row.get("pattern", "")).strip()
        if not pattern: continue
        cleaned_rules.append({
            "name": str(row.get("name", "")).strip(),
            "match_mode": str(row.get("match_mode", "prefix")).strip() or "prefix",
            "pattern": pattern,
            "forbid_aisles": [str(x).strip() for x in row.get("forbid_aisles", []) if str(x).strip()],
            "prefer_aisles": [str(x).strip() for x in row.get("prefer_aisles", []) if str(x).strip()],
        })
    br["aisle_rules"] = cleaned_rules
    return cfg

def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return deepcopy(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_config(raw)

def save_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = normalize_config(config)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    return cfg

# ==========================================
# 3. Result Store
# ==========================================
TABLE_MAP = {
    "关联规则": ("csv", None),
    "调仓建议": ("xlsx", "adjustments"),
    "移动清单": ("xlsx", "move_list"),
    "跳过记录": ("xlsx", "skips"),
    "关联组任务": ("xlsx", "group_tasks"),
    "物料-关联组映射": ("xlsx", "group_material_map"),
}

def clear_cache() -> None:
    _load_cached.cache_clear()

def _file_for(table_name: str) -> tuple[Path, str | None]:
    kind, sheet = TABLE_MAP[table_name]
    if kind == "csv": return OUTPUT_DIR / RULES_CSV_NAME, sheet
    return OUTPUT_DIR / RESULTS_XLSX_NAME, sheet

@lru_cache(maxsize=32)
def _load_cached(path_text: str, mtime_ns: int, sheet: str | None) -> pd.DataFrame:
    path = Path(path_text)
    if path.suffix.lower() == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try: return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError: continue
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)

def load_table(table_name: str) -> pd.DataFrame:
    if table_name not in TABLE_MAP: raise ValueError(f"未知结果表: {table_name}")
    path, sheet = _file_for(table_name)
    if not path.exists(): return pd.DataFrame()
    stat = path.stat()
    return _load_cached(str(path), stat.st_mtime_ns, sheet).copy()

def get_page(table_name: str, page: int = 1, page_size: int = 100, search: str = "") -> tuple[pd.DataFrame, int, int, int]:
    df = load_table(table_name)
    if df.empty: return df, 1, 1, 0
    term = str(search or "").strip().lower()
    if term:
        mask = pd.Series(False, index=df.index)
        for col in df.columns:
            mask = mask | df[col].astype(str).str.lower().str.contains(term, regex=False, na=False)
        df = df[mask]
    total = len(df)
    page_size = max(20, min(int(page_size or 100), 500))
    pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(int(page or 1), pages))
    start = (page - 1) * page_size
    view = df.iloc[start:start + page_size].copy()
    return view.fillna(""), page, pages, total

# ==========================================
# 4. Visualization (Matplotlib & Plotly)
# ==========================================
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

def _natural_aisle_key(value: object) -> tuple[int, str]:
    text = str(value)
    try: return (0, f"{int(text):08d}")
    except Exception: return (1, text)

def _group_key(value: object) -> str:
    if pd.isna(value): return ""
    if isinstance(value, float) and value.is_integer(): return str(int(value))
    return str(value)

def _group_colors(groups: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    items = sorted({g for g in groups if g}, key=_natural_aisle_key)
    if not items: return {}
    colors: dict[str, tuple[float, float, float]] = {}
    golden = 0.618033988749895
    for i, group in enumerate(items):
        hue = (0.08 + i * golden) % 1.0
        sat = 0.58 if i % 2 == 0 else 0.68
        val = 0.84 if i % 3 else 0.74
        colors[group] = tuple(hsv_to_rgb((hue, sat, val)))
    return colors

def _tooltip(row: pd.Series, moved: bool) -> str:
    fields = [
        ("物料", row.get("物料", "")), ("关联组", row.get("关联组", "")), ("库位", row.get("仓位", "")),
        ("通道", row.get("aisle", "")), ("Bay", row.get("bay", "")), ("层", row.get("level", "")),
        ("子库位", row.get("sub_bay", "")), ("优先级序号", row.get("优先级序号", "")),
        ("前端优先级", row.get("前端优先级", "")), ("关联组并集频率", row.get("关联组并集频率", "")),
        ("关联组并集订单数", row.get("关联组并集订单数", "")), ("锚点物料", row.get("锚点物料", "")),
        ("本次调仓涉及", "是" if moved else "否"),
    ]
    return "\n".join(f"{name}: {value}" for name, value in fields if not pd.isna(value) and str(value) != "")

def _inject_svg_titles(svg_path: Path, tooltips: dict[str, str]) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    for gid, text in tooltips.items():
        target = root.find(f".//*[@id='{gid}']")
        if target is None: continue
        target.set("class", (target.get("class", "") + " slot-point").strip())
        title = ET.Element(f"{{{SVG_NS}}}title")
        title.text = text
        target.insert(0, title)
    style = ET.Element(f"{{{SVG_NS}}}style")
    style.text = ".slot-point { cursor: crosshair; } .slot-point:hover { filter: brightness(0.82); } .slot-point:hover path { stroke: #111827 !important; stroke-width: 1.8 !important; }"
    root.insert(0, style)
    root.set("width", "100%")
    root.set("height", "auto")
    root.set("preserveAspectRatio", "xMidYMid meet")
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)

def _offset_codes(series: pd.Series) -> dict[str, int]:
    vals = sorted(series.fillna("").astype(str).unique().tolist())
    center = (len(vals) - 1) / 2.0
    return {value: int(i - center) for i, value in enumerate(vals)}

def render_layout_svg(df: pd.DataFrame, output_path: Path, *, title: str, color_map: dict[str, tuple[float, float, float]] | None = None, moved_materials: set[str] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    moved_materials = {str(x) for x in (moved_materials or set())}
    work = df.copy()
    if "关联组" in work.columns: work = work[work["关联组"].notna()].copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        ax.text(0.5, 0.5, "暂无关联组布局数据", ha="center", va="center", fontsize=16, color="#6b7280")
        ax.set_axis_off()
        fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    work["aisle"] = work["aisle"].astype(str).str.zfill(2)
    work["bay"] = pd.to_numeric(work["bay"], errors="coerce")
    work = work.dropna(subset=["bay"]).copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(11.5, 7.2))
        ax.text(0.5, 0.5, "关联组物料缺少有效库位坐标", ha="center", va="center", fontsize=16, color="#6b7280")
        ax.set_axis_off()
        fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return output_path

    aisles = sorted(work["aisle"].dropna().unique().tolist(), key=_natural_aisle_key)
    aisle_x = {aisle: i for i, aisle in enumerate(aisles)}
    work["_x"] = work["aisle"].map(aisle_x).astype(float)
    if "level" in work.columns:
        level_codes = _offset_codes(work["level"])
        work["_x"] += work["level"].fillna("").astype(str).map(level_codes).fillna(0) * 0.035
    if "sub_bay" in work.columns:
        sub_codes = _offset_codes(work["sub_bay"])
        work["_x"] += work["sub_bay"].fillna("").astype(str).map(sub_codes).fillna(0) * 0.020

    groups = [_group_key(v) for v in work["关联组"]]
    if color_map is None: color_map = _group_colors(groups)

    width = max(10.5, min(18.0, 7.5 + len(aisles) * 0.42))
    fig, ax = plt.subplots(figsize=(width, 7.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfa")

    tooltips: dict[str, str] = {}
    slot_w, slot_h = 0.16, 0.62
    for idx, (_, row) in enumerate(work.iterrows()):
        material, group = str(row.get("物料", "")), _group_key(row.get("关联组"))
        moved = material in moved_materials
        x, y = float(row["_x"]), float(row["bay"])
        rect = Rectangle((x - slot_w / 2, y - slot_h / 2), slot_w, slot_h, facecolor=color_map.get(group, (0.55, 0.58, 0.62)), edgecolor="#dc2626" if moved else "#ffffff", linewidth=2.0 if moved else 0.75, alpha=0.96, zorder=4 if moved else 3)
        gid = f"slot-{idx}"
        rect.set_gid(gid)
        ax.add_patch(rect)
        tooltips[gid] = _tooltip(row, moved)
        if str(row.get("锚点物料", "")) == material:
            anchor = Circle((x, y), radius=0.036, facecolor="#111827", edgecolor="none", zorder=5)
            anchor.set_gid(f"anchor-{idx}")
            ax.add_patch(anchor)

    ax.set_title(title, loc="left", fontsize=17, fontweight="semibold", color="#111827", pad=16)
    ax.text(0.0, 1.015, "方块颜色=关联组；红色描边=本次调仓涉及物料；黑色圆点=锚点物料；鼠标悬浮可查看明细", transform=ax.transAxes, fontsize=9.5, color="#6b7280", va="bottom")
    ax.set_xlabel("Aisle / 通道", color="#374151")
    ax.set_ylabel("Bay", color="#374151")
    ax.set_xticks(list(range(len(aisles))))
    ax.set_xticklabels(aisles)
    bay_min, bay_max = float(work["bay"].min()), float(work["bay"].max())
    pad_y = max(0.8, (bay_max - bay_min) * 0.025)
    ax.set_ylim(bay_max + pad_y, bay_min - pad_y)
    ax.grid(True, which="major", axis="both", color="#e5e7eb", linewidth=0.75, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values(): spine.set_color("#d1d5db")
    ax.tick_params(colors="#4b5563", labelsize=9)
    ax.set_xlim(-0.55, max(len(aisles) - 0.45, 0.55))

    unique_groups = sorted({g for g in groups if g}, key=_natural_aisle_key)
    if len(unique_groups) <= 14:
        handles = [Rectangle((0, 0), 1, 1, facecolor=color_map[g], edgecolor="white", label=f"组 {g}") for g in unique_groups]
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, title="关联组", title_fontsize=9)
            fig.subplots_adjust(right=0.84)

    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _inject_svg_titles(output_path, tooltips)
    return output_path

def render_layout_pair(before_df: pd.DataFrame, after_df: pd.DataFrame, before_path: Path, after_path: Path, moved_materials: set[str] | None = None) -> tuple[Path, Path]:
    all_groups: list[str] = []
    for frame in (before_df, after_df):
        if "关联组" in frame.columns: all_groups.extend(_group_key(v) for v in frame["关联组"].dropna())
    color_map = _group_colors(all_groups)
    render_layout_svg(before_df, before_path, title="调仓前关联组布局", color_map=color_map, moved_materials=moved_materials)
    render_layout_svg(after_df, after_path, title="调仓后关联组布局", color_map=color_map, moved_materials=moved_materials)
    return before_path, after_path

def svg_as_html(path: Path) -> str:
    if not path.exists(): return '<div>暂无 SVG 布局，请先运行调仓计算。</div>'
    svg = path.read_text(encoding="utf-8")
    svg = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg, count=1)
    svg = re.sub(r"^\s*<!DOCTYPE[^>]*>\s*", "", svg, count=1, flags=re.IGNORECASE)
    return f'<div style="width:100%; overflow:auto;">{svg}</div>'

def _rgb_css(color: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, int(round(c * 255)))) for c in color)
    return f"rgb({r},{g},{b})"

def empty_layout_plot(message: str = "暂无布局数据，请先运行调仓计算。") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font={"size": 16, "color": "#6b7280"})
    fig.update_layout(template="plotly_white", height=620, margin={"l": 55, "r": 30, "t": 55, "b": 55}, xaxis={"visible": False}, yaxis={"visible": False})
    return fig

def build_layout_plot(df: pd.DataFrame, *, title: str, color_map: dict[str, tuple[float, float, float]] | None = None, moved_materials: set[str] | None = None) -> go.Figure:
    moved_materials = {str(x) for x in (moved_materials or set())}
    work = df.copy()
    if "关联组" in work.columns: work = work[work["关联组"].notna()].copy()
    if work.empty: return empty_layout_plot("暂无关联组布局数据")

    work["aisle"] = work["aisle"].astype(str).str.zfill(2)
    work["bay"] = pd.to_numeric(work["bay"], errors="coerce")
    work = work.dropna(subset=["bay"]).copy()
    if work.empty: return empty_layout_plot("关联组物料缺少有效库位坐标")

    aisles = sorted(work["aisle"].dropna().unique().tolist(), key=_natural_aisle_key)
    aisle_x = {aisle: i for i, aisle in enumerate(aisles)}
    work["_x"] = work["aisle"].map(aisle_x).astype(float)
    if "level" in work.columns:
        level_codes = _offset_codes(work["level"])
        work["_x"] += work["level"].fillna("").astype(str).map(level_codes).fillna(0) * 0.035
    if "sub_bay" in work.columns:
        sub_codes = _offset_codes(work["sub_bay"])
        work["_x"] += work["sub_bay"].fillna("").astype(str).map(sub_codes).fillna(0) * 0.020

    work["_group"] = work["关联组"].map(_group_key)
    if color_map is None: color_map = _group_colors(work["_group"].tolist())

    hover_fields = [("物料", "物料"), ("关联组", "关联组"), ("库位", "仓位"), ("通道", "aisle"), ("Bay", "bay"), ("层", "level"), ("子库位", "sub_bay"), ("优先级序号", "优先级序号"), ("前端优先级", "前端优先级"), ("关联组并集频率", "关联组并集频率"), ("关联组并集订单数", "关联组并集订单数"), ("锚点物料", "锚点物料")]
    def make_hover(row: pd.Series) -> str:
        lines = [f"<b>{label}</b>: {value}" for label, col in hover_fields if not pd.isna(value := row.get(col, "")) and str(value) != ""]
        lines.append(f"<b>本次调仓涉及</b>: {'是' if str(row.get('物料', '')) in moved_materials else '否'}")
        return "<br>".join(lines)

    work["_hover"] = work.apply(make_hover, axis=1)
    fig = go.Figure()
    groups = sorted(work["_group"].unique().tolist(), key=_natural_aisle_key)
    for group in groups:
        part = work[work["_group"] == group]
        fig.add_trace(go.Scatter(x=part["_x"], y=part["bay"], mode="markers", name=f"组 {group}", legendgroup=f"group-{group}", marker={"symbol": "square", "size": 10, "color": _rgb_css(color_map.get(group, (0.55, 0.58, 0.62))), "line": {"color": "rgba(255,255,255,0.8)", "width": 0.8}}, text=part["_hover"], hovertemplate="%{text}<extra></extra>"))

    if moved_materials:
        moved = work[work["物料"].astype(str).isin(moved_materials)]
        if not moved.empty:
            fig.add_trace(go.Scatter(x=moved["_x"], y=moved["bay"], mode="markers", name="本次调仓涉及", marker={"symbol": "square-open", "size": 15, "color": "#dc2626", "line": {"color": "#dc2626", "width": 2.2}}, hoverinfo="skip", showlegend=True))

    anchor_mask = pd.Series(False, index=work.index)
    if "锚点物料" in work.columns: anchor_mask = work["锚点物料"].astype(str) == work["物料"].astype(str)
    anchors = work[anchor_mask]
    if not anchors.empty:
        fig.add_trace(go.Scatter(x=anchors["_x"], y=anchors["bay"], mode="markers", name="锚点物料", marker={"symbol": "circle", "size": 5, "color": "#111827"}, hoverinfo="skip", showlegend=True))

    bay_min, bay_max = float(work["bay"].min()), float(work["bay"].max())
    pad_y = max(0.8, (bay_max - bay_min) * 0.025)
    fig.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"}, template="plotly_white", height=650, hovermode="closest", dragmode="pan", margin={"l": 60, "r": 35, "t": 70, "b": 60}, legend={"title": {"text": "关联组 / 标记"}, "itemsizing": "constant"},
        xaxis={"title": "Aisle / 通道", "tickmode": "array", "tickvals": list(range(len(aisles))), "ticktext": aisles, "range": [-0.55, max(len(aisles) - 0.45, 0.55)], "showgrid": True, "gridcolor": "#e5e7eb", "zeroline": False},
        yaxis={"title": "Bay", "range": [bay_max + pad_y, bay_min - pad_y], "showgrid": True, "gridcolor": "#e5e7eb", "zeroline": False},
    )
    return fig

def build_layout_plot_pair(before_df: pd.DataFrame, after_df: pd.DataFrame, *, moved_materials: set[str] | None = None) -> tuple[go.Figure, go.Figure]:
    all_groups: list[str] = []
    for frame in (before_df, after_df):
        if "关联组" in frame.columns: all_groups.extend(_group_key(v) for v in frame["关联组"].dropna())
    color_map = _group_colors(all_groups)
    return build_layout_plot(before_df, title="调仓前关联组布局", color_map=color_map, moved_materials=moved_materials), build_layout_plot(after_df, title="调仓后关联组布局", color_map=color_map, moved_materials=moved_materials)

# ==========================================
# 5. Algorithms (FP-Growth & Reslotting)
# ==========================================
def run_fpgrowth(df_ship, df_lx03, min_support=0.015, valid_aisles=[]):
    start_time = time.perf_counter()
    print('[步骤1/7]：开始运行 fpgrowth')
    ship_df_analysis = df_ship[['TO number', 'Material']].copy()
    ship_df_analysis = pd.merge(ship_df_analysis, df_lx03, left_on='Material', right_on='物料', how='left')
    ship_df_analysis['Material'] = ship_df_analysis['Material'].astype(str) + ' @ ' + ship_df_analysis['仓位'].astype(str)
    ship_df_analysis.dropna(subset=['仓位'], inplace=True)
    if valid_aisles:
        ship_df_analysis = ship_df_analysis[pd.to_numeric(ship_df_analysis['仓位'].str[:2], errors='coerce').isin(valid_aisles)]
    
    ship_onehot = pd.crosstab(ship_df_analysis['TO number'], ship_df_analysis['Material']).astype(bool)
    n_tx = len(ship_onehot)
    ship_onehot = ship_onehot.astype(int).loc[:, ship_onehot.sum(axis=0) > 0]
    frequent_itemsets_fpgrowth = fpgrowth(ship_onehot, min_support=min_support, use_colnames=True)
    if frequent_itemsets_fpgrowth.empty: return None
    
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric='confidence', min_threshold=0.5)
    if rules_fpgrowth.empty: return rules_fpgrowth
    
    rules_fpgrowth = rules_fpgrowth.copy()
    rules_fpgrowth['union_items'] = pd.Series([frozenset(a) | frozenset(c) for a, c in zip(rules_fpgrowth['antecedents'], rules_fpgrowth['consequents'])], index=rules_fpgrowth.index, dtype=object)
    ship_bool = ship_onehot.astype(bool)
    col_arrays = {col: ship_bool[col].to_numpy(copy=False) for col in ship_bool.columns}
    unique_unions = rules_fpgrowth['union_items'].drop_duplicates()
    union_count_map = {}
    for union_items in unique_unions:
        arrs = [col_arrays[item] for item in union_items if item in col_arrays]
        if not arrs: union_count_map[union_items] = 0
        elif len(arrs) == 1: union_count_map[union_items] = int(arrs[0].sum())
        else: union_count_map[union_items] = int(np.logical_or.reduce(arrs).sum())
    
    rules_fpgrowth['group_union_order_count'] = rules_fpgrowth['union_items'].map(union_count_map)
    rules_fpgrowth['group_union_support'] = rules_fpgrowth['group_union_order_count'] / n_tx if n_tx else 0.0
    rules_fpgrowth.drop(columns=['union_items'], inplace=True)
    print(f'[步骤7/7]完成：输出规则数量 = {len(rules_fpgrowth)}，总耗时 = {time.perf_counter() - start_time:.2f} 秒')
    return rules_fpgrowth

EMPTY_MARKER = '<<空的>>'
SKIP_REASON_LABELS = {
    'already_grouped': '该关联组已经集中在目标窗口中', 'budget_exhausted': '已达到本次调仓预算上限',
    'insufficient_active_materials': '剔除缺失库位或保留库位后，规则内可参与物料不足 2 个',
    'missing_bin': '物料当前没有可用库位', 'no_better_candidate': '存在候选库位或交换对象，但估计净收益不为正',
    'no_candidate': '没有找到满足约束的空位或交换对象', 'no_common_allowed_aisle': '该关联组没有共同允许进入的通道',
    'no_target_window': '没有找到合适的前端连续窗口', 'reserved_bin': '物料所在库位被配置为保留库位，不参与调仓',
    'target_aisle_forbidden': '物料受通道约束限制，不能进入目标通道',
    'swap_target_aisle_forbidden': '被换出的物料受通道约束限制，不能进入交换后的目标通道',
}
SCOPE_LABELS = {'rule': '规则级', 'material': '物料级'}

def parse_rule_materials(value) -> List[dict]:
    if pd.isna(value): return []
    text = str(value).strip()
    if not text: return []
    items = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk: continue
        if ' @ ' in chunk: material, bin_name = chunk.split(' @ ', 1)
        elif '@' in chunk: material, bin_name = chunk.split('@', 1)
        else: material, bin_name = chunk, None
        items.append({'material': material.strip(), 'bin': None if bin_name is None else bin_name.strip()})
    return items

def _dedupe_materials(items: Sequence[dict]) -> Tuple[List[str], List[str]]:
    materials, bins, seen = [], [], set()
    for item in items:
        material = item.get('material')
        if not material or material in seen: continue
        seen.add(material); materials.append(material); bins.append(item.get('bin'))
    return materials, bins

def _safe_rank(series: pd.Series) -> pd.Series:
    if series.isna().all(): return pd.Series(np.ones(len(series)), index=series.index, dtype=float)
    filled = series.fillna(series.min() if series.notna().any() else 0)
    return filled.rank(method='average', pct=True)

def _first_existing_numeric(df: pd.DataFrame, candidates: Sequence[str], default: float = 0.0) -> pd.Series:
    for col in candidates:
        if col in df.columns: return pd.to_numeric(df[col], errors='coerce')
    return pd.Series(default, index=df.index, dtype=float)

def _normalize_material_code(material: str) -> str: return re.sub(r'[^A-Z0-9]', '', str(material).upper())
def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ch_a, ch_b in zip(a, b):
        if ch_a != ch_b: break
        n += 1
    return n
def _common_suffix_len(a: str, b: str) -> int: return _common_prefix_len(a[::-1], b[::-1])

def _material_confusion_penalty(material_a: Optional[str], material_b: Optional[str]) -> float:
    if not material_a or not material_b: return 0.0
    a, b = _normalize_material_code(material_a), _normalize_material_code(material_b)
    if not a or not b: return 0.0
    if a == b: return 10.0
    prefix_len, suffix_len = _common_prefix_len(a, b), _common_suffix_len(a, b)
    same_pos = sum(ch_a == ch_b for ch_a, ch_b in zip(a, b))
    max_len = max(len(a), len(b))
    overlap_ratio = same_pos / max_len if max_len else 0.0
    diff_count = abs(len(a) - len(b))
    if len(a) == len(b): diff_count += sum(ch_a != ch_b for ch_a, ch_b in zip(a, b))
    near_identical = 1.0 if diff_count <= 2 else 0.0
    return 0.45 * min(prefix_len, 6) + 0.70 * min(suffix_len, 4) + 2.20 * overlap_ratio + 1.80 * near_identical

def _nearest_group_neighbor(window_bins: Sequence[str], occupant_map: Dict[str, Optional[str]], pos: int, direction: int) -> Optional[str]:
    idx = pos + direction
    while 0 <= idx < len(window_bins):
        occupant = occupant_map.get(window_bins[idx])
        if occupant not in (None, EMPTY_MARKER): return occupant
        idx += direction
    return None

def _build_group_assignment_plan(state: "State", group_materials: Sequence[str], window_bins: Sequence[str], target_aisle: str, occupant_map: Dict[str, Optional[str]], *, quality_penalty_weight: float) -> Tuple[List[str], List[str]]:
    mats_already_in_window = {occ for occ in occupant_map.values() if occ in group_materials}
    target_bins_to_fill = [bin_name for bin_name in window_bins if occupant_map.get(bin_name) not in group_materials]
    candidate_materials = [m for m in group_materials if m not in mats_already_in_window]
    if not target_bins_to_fill or not candidate_materials: return target_bins_to_fill, candidate_materials
    
    heat_series = pd.Series({material: float(state.material_weight.get(material, 0.0)) for material in candidate_materials}, dtype=float)
    heat_rank = _safe_rank(heat_series) if not heat_series.empty else pd.Series(dtype=float)
    source_bay_series = pd.Series({material: float(state.bin_info[state.mat_to_bin[material]]['bay']) for material in candidate_materials}, dtype=float)
    source_bay_rank = _safe_rank(source_bay_series) if not source_bay_series.empty else pd.Series(dtype=float)
    
    pending, ordered_materials, mutable_occupant_map = list(candidate_materials), [], dict(occupant_map)
    total_slots = max(len(target_bins_to_fill), 1)
    for slot_pos, target_bin in enumerate(target_bins_to_fill):
        best_material, best_score, window_pos = None, None, list(window_bins).index(target_bin)
        slot_front_factor = 1.0 - slot_pos / total_slots
        left_neighbor = _nearest_group_neighbor(window_bins, mutable_occupant_map, window_pos, -1)
        right_neighbor = _nearest_group_neighbor(window_bins, mutable_occupant_map, window_pos, +1)
        for material in pending:
            current_bin, current_info = state.mat_to_bin[material], state.bin_info[state.mat_to_bin[material]]
            cross_aisle_bonus = 1.0 if str(current_info['aisle']) != str(target_aisle) else 0.0
            front_pull = max(0.0, float(current_info['bay']) - float(state.bin_info[target_bin]['bay']))
            material_heat, source_depth = float(heat_rank.get(material, 0.0)), float(source_bay_rank.get(material, 0.0))
            confusion_penalty = _material_confusion_penalty(material, left_neighbor) + 0.8 * _material_confusion_penalty(material, right_neighbor)
            score = (2.0 + 1.8 * slot_front_factor) * material_heat + 1.1 * slot_front_factor * source_depth + 1.0 * cross_aisle_bonus + 0.15 * front_pull - quality_penalty_weight * confusion_penalty
            if best_score is None or score > best_score: best_score, best_material = score, material
        ordered_materials.append(best_material)
        mutable_occupant_map[target_bin] = best_material
        pending.remove(best_material)
    return target_bins_to_fill, ordered_materials

def prepare_rule_analysis(df_related_mat: pd.DataFrame, support_weight: float = 0.50, confidence_weight: float = 0.15, lift_weight: float = 0.35, evidence_bonus: float = 0.12, rule_size_penalty_exponent: float = 0.50) -> Tuple[pd.DataFrame, pd.DataFrame]:
    is_valid_rule = (df_related_mat['len_antecedents'] >= 1) & (df_related_mat['len_consequents'] >= 1)
    df_rule_raw = df_related_mat.loc[is_valid_rule].copy()
    if df_rule_raw.empty: return pd.DataFrame(columns=['rule_materials', 'rule_bins', 'rule_weight']), pd.DataFrame()
    
    antecedent_items = [parse_rule_materials(v) for v in df_rule_raw['前项物料']]
    consequent_items = [parse_rule_materials(v) for v in df_rule_raw['后项物料']]
    rule_items = [a + c for a, c in zip(antecedent_items, consequent_items)]
    
    rule_materials, rule_bins, signatures, rule_sizes, valid_mask = [], [], [], [], []
    for items in rule_items:
        mats, bins = _dedupe_materials(items)
        is_valid = len(mats) >= 2 and sum(1 for b in bins if b) >= 2
        valid_mask.append(is_valid); rule_materials.append(mats); rule_bins.append(bins)
        signatures.append('||'.join(sorted(mats))); rule_sizes.append(len(mats))
        
    df_rule_raw = df_rule_raw.loc[valid_mask].copy()
    if df_rule_raw.empty: return pd.DataFrame(columns=['rule_materials', 'rule_bins', 'rule_weight']), pd.DataFrame()
    
    df_rule_raw['rule_materials'] = [x for x, keep in zip(rule_materials, valid_mask) if keep]
    df_rule_raw['rule_bins'] = [x for x, keep in zip(rule_bins, valid_mask) if keep]
    df_rule_raw['rule_signature'] = [x for x, keep in zip(signatures, valid_mask) if keep]
    df_rule_raw['rule_size'] = [x for x, keep in zip(rule_sizes, valid_mask) if keep]
    
    union_support_candidates = ['group_union_support', '关联组并集频率', '关联组订单并集频率']
    union_order_count_candidates = ['group_union_order_count', '关联组并集订单数', '关联组订单并集订单数']
    df_rule_raw['group_union_support'] = _first_existing_numeric(df_rule_raw, union_support_candidates, default=np.nan)
    df_rule_raw['group_union_order_count'] = _first_existing_numeric(df_rule_raw, union_order_count_candidates, default=np.nan)
    
    agg_map = {'rule_materials': 'first', 'rule_bins': 'first', 'rule_size': 'first', 'len_antecedents': 'max', 'len_consequents': 'max', 'group_union_support': 'max', 'group_union_order_count': 'max'}
    for col in ['支持度', '置信度', '提升度', '老仓', '新仓', '同仓库拣货', '同通道拣货']:
        if col in df_rule_raw.columns: agg_map[col] = 'max'
        
    df_rule_analysis = df_rule_raw.groupby('rule_signature', as_index=False).agg(agg_map)
    evidence_counts = df_rule_raw.groupby('rule_signature').size().rename('evidence_count').reset_index()
    df_rule_analysis = df_rule_analysis.merge(evidence_counts, on='rule_signature', how='left')
    
    support_rank = _safe_rank(df_rule_analysis['支持度']) if '支持度' in df_rule_analysis.columns else pd.Series(1.0, index=df_rule_analysis.index)
    confidence_rank = _safe_rank(df_rule_analysis['置信度']) if '置信度' in df_rule_analysis.columns else pd.Series(1.0, index=df_rule_analysis.index)
    lift_rank = _safe_rank(df_rule_analysis['提升度']) if '提升度' in df_rule_analysis.columns else pd.Series(1.0, index=df_rule_analysis.index)
    
    df_rule_analysis['support_rank'] = support_rank
    df_rule_analysis['confidence_rank'] = confidence_rank
    df_rule_analysis['lift_rank'] = lift_rank
    df_rule_analysis['rule_score_base'] = support_weight * support_rank + confidence_weight * confidence_rank + lift_weight * lift_rank
    df_rule_analysis['size_penalty'] = df_rule_analysis['rule_size'].clip(lower=2).pow(-rule_size_penalty_exponent)
    df_rule_analysis['evidence_bonus_factor'] = 1.0 + evidence_bonus * np.log1p(df_rule_analysis['evidence_count'].fillna(1))
    df_rule_analysis['rule_weight'] = df_rule_analysis['rule_score_base'] * df_rule_analysis['size_penalty'] * df_rule_analysis['evidence_bonus_factor']
    
    df_rule_analysis = df_rule_analysis.sort_values(['rule_weight', 'group_union_support', 'support_rank', 'evidence_count', 'rule_size', 'rule_signature'], ascending=[False, False, False, False, True, True]).reset_index(drop=True)
    return df_rule_analysis, df_rule_analysis.loc[df_rule_analysis['rule_size'] == 2].copy()

def _build_rule_rank_map(df_rule_analysis: pd.DataFrame) -> Dict[int, int]:
    if 'rule_weight' in df_rule_analysis.columns and not df_rule_analysis.empty:
        df_rule_rank = df_rule_analysis[['rule_weight', 'support_rank']].copy()
        df_rule_rank['_row_index'] = df_rule_rank.index
        df_rule_rank = df_rule_rank.sort_values(['rule_weight', 'support_rank', '_row_index'], ascending=[False, False, True])
        return {row_idx: rank for rank, row_idx in enumerate(df_rule_rank['_row_index'].tolist())}
    return {row_idx: rank for rank, row_idx in enumerate(df_rule_analysis.index.tolist())}

def _build_material_stats(df_rule_analysis: pd.DataFrame):
    rows = []
    for _, row in df_rule_analysis.iterrows():
        materials = [m for m in row['rule_materials'] if m]
        if not materials: continue
        share = row.get('rule_weight', 1.0) / len(materials)
        for material in materials: rows.append((material, share))
    if rows:
        df_long = pd.DataFrame(rows, columns=['material', 'weight'])
        material_weight = df_long.groupby('material', observed=True)['weight'].sum().to_dict()
    else: material_weight = {}
    rule_rank_map = _build_rule_rank_map(df_rule_analysis)
    material_priority = {}
    for idx, row in df_rule_analysis.iterrows():
        rank = rule_rank_map[idx]
        for material in row['rule_materials']: material_priority[material] = min(material_priority.get(material, rank), rank)
    return material_weight, material_priority, rule_rank_map

def _build_pair_stats(df_rule_analysis: pd.DataFrame, mat_to_bin: Dict[str, str]):
    pair_weight = defaultdict(float)
    for row in df_rule_analysis[['rule_materials', 'rule_weight']].itertuples(index=False):
        mats = [m for m in row.rule_materials if m in mat_to_bin]
        if len(mats) < 2: continue
        mats = list(dict.fromkeys(mats))
        if len(mats) < 2: continue
        denom = math.comb(len(mats), 2)
        share = float(row.rule_weight) / max(1, denom)
        for idx, a in enumerate(mats):
            for b in mats[idx + 1:]:
                key = (a, b) if a <= b else (b, a)
                pair_weight[key] += share
    neighbor_map = defaultdict(dict)
    for (a, b), weight in pair_weight.items(): neighbor_map[a][b] = weight; neighbor_map[b][a] = weight
    return pair_weight, {k: dict(v) for k, v in neighbor_map.items()}

def _build_small_cluster_maps(neighbor_map: Dict[str, Dict[str, float]], materials: Sequence[str], top_k=3, keep_ratio=0.45):
    parent = {m: m for m in materials}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    local_cluster_map = {}
    for material in materials:
        neighbors = neighbor_map.get(material, {})
        if not neighbors: local_cluster_map[material] = [material]; continue
        ranked = sorted(neighbors.items(), key=lambda item: (-item[1], item[0]))
        threshold = ranked[0][1] * keep_ratio
        kept = [n for n, w in ranked if w >= threshold][:top_k]
        local_cluster_map[material] = [material] + kept
        for neighbor in kept: union(material, neighbor)
    component_to_materials = defaultdict(set)
    material_to_component = {}
    for material in materials:
        root = find(material)
        material_to_component[material] = root
        component_to_materials[root].add(material)
    return material_to_component, {root: sorted(mats) for root, mats in component_to_materials.items()}, local_cluster_map

@dataclass
class State:
    bin_info: dict; bin_to_mat: dict; mat_to_bin: dict; empty_bins: set; empty_bins_by_aisle: dict
    all_aisles: list; reserved_bins: set; material_weight: dict; material_priority: dict; finalized_materials: set
    entity_to_bin: dict; bin_to_entity: dict; entity_label: dict; entity_inventory: dict; entity_best_rule_rank: dict
    aisle_rules: list; aisle_priority: list; protected_bin_owner: dict = None; protected_material_owner: dict = None

def _compile_bin_patterns(patterns):
    compiled = []
    for pattern in patterns or []:
        text = str(pattern).strip().upper()
        if not text: continue
        regex_text = re.escape(text).replace(r'*', '.*').replace('X', '.')
        compiled.append(re.compile(f'^{regex_text}$'))
    return compiled

def _match_reserved_bins(bins, reserved_bin_patterns):
    compiled = _compile_bin_patterns(reserved_bin_patterns)
    if not compiled: return set()
    return {bin_name for bin_name in bins if any(regex.match(str(bin_name).upper()) for regex in compiled)}

def _material_matches_rule(material, rule):
    match_mode = rule.get('match_mode', 'prefix')
    pattern = str(rule.get('pattern', '')).strip()
    if not pattern: return False
    if match_mode == 'exact': return material == pattern
    if match_mode == 'contains': return pattern in material
    if match_mode == 'regex': return re.search(pattern, material) is not None
    return material.startswith(pattern)

def _get_material_aisle_policy(material, aisle_rules):
    forbid_aisles, prefer_aisles = set(), []
    for rule in aisle_rules:
        if not _material_matches_rule(material, rule): continue
        forbid_aisles.update(str(aisle) for aisle in rule.get('forbid_aisles', []))
        for aisle in rule.get('prefer_aisles', []):
            aisle = str(aisle)
            if aisle not in prefer_aisles: prefer_aisles.append(aisle)
    return {'forbid_aisles': forbid_aisles, 'prefer_aisles': [aisle for aisle in prefer_aisles if aisle not in forbid_aisles]}

def _is_aisle_allowed_for_material(material, aisle, aisle_rules):
    return str(aisle) not in _get_material_aisle_policy(material, aisle_rules)['forbid_aisles']

def _get_common_allowed_aisles(materials, state: State):
    return [aisle for aisle in state.all_aisles if all(_is_aisle_allowed_for_material(material, aisle, state.aisle_rules) for material in materials)]

def _get_group_preference_stats(materials: Sequence[str], aisle: str, aisle_rules) -> Tuple[int, int, float]:
    aisle = str(aisle)
    prefer_hit_count, prefer_defined_count = 0, 0
    for material in materials:
        prefer_aisles = [str(a) for a in _get_material_aisle_policy(material, aisle_rules).get('prefer_aisles', [])]
        if prefer_aisles:
            prefer_defined_count += 1
            if aisle in prefer_aisles: prefer_hit_count += 1
    prefer_miss_count = prefer_defined_count - prefer_hit_count
    prefer_hit_ratio = prefer_hit_count / prefer_defined_count if prefer_defined_count else 0.0
    return prefer_hit_count, prefer_miss_count, prefer_hit_ratio

def _aisle_numeric(aisle: str) -> int:
    try: return int(str(aisle))
    except Exception:
        digits = ''.join(ch for ch in str(aisle) if ch.isdigit())
        return int(digits) if digits else 0

def _front_rank_for_aisle(aisle: str, aisle_priority: Sequence[str]) -> float:
    aisle = str(aisle)
    priority_list = [str(x) for x in (aisle_priority or [])]
    if aisle in priority_list: return float(priority_list.index(aisle))
    return float(len(priority_list)) + _aisle_numeric(aisle) / 100.0

def _travel_cost(state: State, bin_a: str, bin_b: str) -> float:
    if bin_a not in state.bin_info or bin_b not in state.bin_info: return float('inf')
    a, b = state.bin_info[bin_a], state.bin_info[bin_b]
    bay_gap = abs(float(a['bay']) - float(b['bay'])) / 2.0
    if a['aisle'] == b['aisle']: return bay_gap
    return 6.0 + abs(_aisle_numeric(a['aisle']) - _aisle_numeric(b['aisle'])) * 1.5 + bay_gap * 0.5

def _distance(state: State, bin_a: str, bin_b: str) -> float: return _travel_cost(state, bin_a, bin_b)

def _sorted_bins_for_aisle(state: State, aisle: str) -> List[str]:
    bins = [bin_name for bin_name, info in state.bin_info.items() if str(info['aisle']) == str(aisle) and bin_name not in state.reserved_bins]
    return sorted(bins, key=lambda bin_name: (float(state.bin_info[bin_name]['bay']), str(state.bin_info[bin_name]['level']), str(state.bin_info[bin_name]['sub_bay']), str(bin_name)))

def _iter_windows(items: Sequence[str], size: int) -> List[List[str]]:
    if not items: return []
    size = max(1, min(int(size), len(items)))
    if size >= len(items): return [list(items)]
    return [list(items[idx: idx + size]) for idx in range(0, len(items) - size + 1)]

def _representative_material(state: State, materials: Sequence[str]) -> Optional[str]:
    if not materials: return None
    return min(materials, key=lambda material: (-state.material_weight.get(material, 0.0), state.material_priority.get(material, 10**9), str(state.mat_to_bin.get(material, '')), str(material)))

def _init_entity_maps(df_bins, df_bin_settings):
    initial_bin_to_mat = df_bins.set_index('仓位')['物料'].to_dict()
    s_setting_mat = df_bin_settings['to_bin_material']
    if not s_setting_mat.index.is_unique: s_setting_mat = s_setting_mat[~s_setting_mat.index.duplicated(keep='first')]
    bin_setting_mat_map = s_setting_mat.to_dict()
    bin_inv = df_bins.set_index('仓位')['总库存量'].to_dict() if '总库存量' in df_bins.columns else {}
    entity_to_bin, bin_to_entity, entity_label, entity_inventory = {}, {}, {}, {}
    for bin_name, material in initial_bin_to_mat.items():
        if material == EMPTY_MARKER:
            entity_id = f'EMPTY@{bin_name}'
            setting_mat = bin_setting_mat_map.get(bin_name)
            label = f'EMPTY_BIN@{bin_name}' if pd.isna(setting_mat) or str(setting_mat).strip() in ('', EMPTY_MARKER) else str(setting_mat)
        else:
            entity_id, label = f'MAT@{material}', material
        if entity_id in entity_to_bin: entity_id = f'{entity_id}@{bin_name}'
        entity_to_bin[entity_id] = bin_name; bin_to_entity[bin_name] = entity_id; entity_label[entity_id] = label; entity_inventory[entity_id] = bin_inv.get(bin_name, np.nan)
    return {'initial_bin_to_mat': initial_bin_to_mat, 'entity_to_bin': entity_to_bin, 'bin_to_entity': bin_to_entity, 'entity_label': entity_label, 'entity_inventory': entity_inventory, 'entity_best_rule_rank': {}}

def _mark_entity_rule_rank(state: State, entity_id, rule_rank):
    old_rank = state.entity_best_rule_rank.get(entity_id)
    if old_rank is None or rule_rank < old_rank: state.entity_best_rule_rank[entity_id] = rule_rank

def _append_skip(rows, scope, rule_index, group_id, rule_materials, competition_materials, anchor_material, target_aisle, materials, pending_materials, reason):
    rows.append({'scope': scope, 'rule_index': rule_index, 'group_id': group_id, 'rule_materials': ', '.join(rule_materials) if isinstance(rule_materials, list) else rule_materials, 'competition_materials': ', '.join(competition_materials) if isinstance(competition_materials, list) else competition_materials, 'anchor_material': anchor_material, 'target_aisle': target_aisle, 'materials': ', '.join(materials) if isinstance(materials, list) else materials, 'pending_materials': ', '.join(pending_materials) if isinstance(pending_materials, list) else pending_materials, 'reason': reason, 'reason_label': SKIP_REASON_LABELS.get(reason, reason), 'scope_label': SCOPE_LABELS.get(scope, scope)})

def _build_move_list(state: State, initial_entity_to_bin):
    rows = []
    for entity_id, old_bin in initial_entity_to_bin.items():
        new_bin = state.entity_to_bin.get(entity_id)
        if pd.notna(new_bin) and new_bin != old_bin:
            rows.append({'物料': state.entity_label[entity_id], '原库位': old_bin, '新库位': new_bin, '总库存量': state.entity_inventory.get(entity_id), '_rule_rank': state.entity_best_rule_rank.get(entity_id)})
    df_move_list = pd.DataFrame(rows)
    if df_move_list.empty: return df_move_list
    old_to_new = dict(zip(df_move_list['原库位'], df_move_list['新库位']))
    old_to_rule_rank = dict(zip(df_move_list['原库位'], df_move_list['_rule_rank']))
    adjacency = {}
    for old_bin, new_bin in old_to_new.items(): adjacency.setdefault(old_bin, set()).add(new_bin); adjacency.setdefault(new_bin, set()).add(old_bin)
    visited, components = set(), []
    for node in adjacency:
        if node in visited: continue
        stack, component = [node], []
        visited.add(node)
        while stack:
            current = stack.pop(); component.append(current)
            for nxt in adjacency.get(current, set()):
                if nxt not in visited: visited.add(nxt); stack.append(nxt)
        components.append(component)
    comp_priority, old_bin_to_comp = {}, {}
    for comp_id, comp_nodes in enumerate(components):
        ranks = [old_to_rule_rank[node] for node in comp_nodes if node in old_to_rule_rank and pd.notna(old_to_rule_rank[node])]
        comp_priority[comp_id] = min(ranks) if ranks else 10**9
        for node in comp_nodes:
            if node in old_to_new: old_bin_to_comp[node] = comp_id
    sorted_comp_ids = sorted(comp_priority, key=lambda cid: (comp_priority[cid], min(str(item) for item in components[cid])))
    comp_to_group_seq = {cid: seq for seq, cid in enumerate(sorted_comp_ids)}
    df_move_list['调仓组'] = df_move_list['原库位'].map(old_bin_to_comp).map(comp_to_group_seq).astype('Int64')
    return df_move_list.sort_values(['调仓组', '原库位']).reset_index(drop=True)[['调仓组', '物料', '原库位', '新库位', '总库存量']]

def _format_outputs(df_adjustments, df_skips, df_move_list, df_group_tasks=None):
    group_id_source = []
    if df_group_tasks is not None and not df_group_tasks.empty and 'group_id' in df_group_tasks.columns: group_id_source.extend(df_group_tasks['group_id'].dropna().tolist())
    if not df_adjustments.empty and 'group_id' in df_adjustments.columns: group_id_source.extend(df_adjustments['group_id'].dropna().tolist())
    if not df_skips.empty and 'group_id' in df_skips.columns: group_id_source.extend(df_skips['group_id'].dropna().tolist())
    if group_id_source:
        group_id_map = {group_id: seq for seq, group_id in enumerate(dict.fromkeys(group_id_source))}
        if df_group_tasks is not None and not df_group_tasks.empty: df_group_tasks['group_id'] = df_group_tasks['group_id'].map(group_id_map).astype('Int64')
        if not df_adjustments.empty: df_adjustments['group_id'] = df_adjustments['group_id'].map(group_id_map).astype('Int64')
        if not df_skips.empty: df_skips['group_id'] = df_skips['group_id'].map(group_id_map).astype('Int64')
    adjustment_cols = {'rule_index': '关联规则索引', 'group_id': '关联组', 'rule_materials': '规则物料', 'competition_materials': '竞争组物料', 'anchor_material': '锚点物料', 'anchor_bin': '锚点库位', 'target_aisle': '目标通道', 'material_mover': '移动物料', 'to_bin_material': '目标库位物料', 'from_bin': '原库位', 'to_bin': '目标库位', 'move_type': '移动类型', 'swap_material': '交换物料', 'old_distance': '调整前距离', 'new_distance': '调整后距离', 'mover_gain': '移动物料收益', 'swap_gain': '交换物料收益', 'move_penalty': '执行惩罚', 'net_gain': '估计净收益', 'front_priority_score': '前端优先级', 'group_union_support': '关联组并集频率', 'group_union_order_count': '关联组并集订单数', 'window_avg_bay': '目标窗口平均bay', 'window_prefer_hit_count': '目标通道偏好命中物料数', 'window_prefer_miss_count': '目标通道偏好未命中物料数', 'window_prefer_hit_ratio': '目标通道偏好命中率'}
    skip_cols = {'scope_label': '跳过层级', 'rule_index': '关联规则索引', 'group_id': '关联组', 'rule_materials': '规则物料', 'competition_materials': '竞争组物料', 'anchor_material': '锚点物料', 'target_aisle': '目标通道', 'materials': '涉及物料', 'pending_materials': '待处理物料', 'reason': '原因代码', 'reason_label': '原因说明'}
    if df_adjustments.empty: df_adjustments = pd.DataFrame(columns=list(adjustment_cols.values()))
    else: df_adjustments = df_adjustments.reindex(columns=list(adjustment_cols.keys())).rename(columns=adjustment_cols)
    if df_skips.empty: df_skips = pd.DataFrame(columns=list(skip_cols.values()))
    else: df_skips = df_skips.reindex(columns=list(skip_cols.keys())).rename(columns=skip_cols)
    return df_adjustments, df_skips, df_move_list, df_group_tasks

def _apply_material_move(state: State, mover_material: str, target_bin: str, move_type: str, rule_rank: Optional[int], swap_material: Optional[str] = None) -> None:
    old_bin = state.mat_to_bin[mover_material]
    mover_entity = state.bin_to_entity[old_bin]
    if move_type == 'empty':
        empty_entity = state.bin_to_entity[target_bin]
        state.bin_to_mat[target_bin] = mover_material; state.mat_to_bin[mover_material] = target_bin
        state.bin_to_mat[old_bin] = EMPTY_MARKER; state.empty_bins.add(old_bin); state.empty_bins_by_aisle.setdefault(state.bin_info[old_bin]['aisle'], set()).add(old_bin)
        state.empty_bins.discard(target_bin); state.empty_bins_by_aisle.setdefault(state.bin_info[target_bin]['aisle'], set()).discard(target_bin)
        state.bin_to_entity[target_bin] = mover_entity; state.entity_to_bin[mover_entity] = target_bin
        state.bin_to_entity[old_bin] = empty_entity; state.entity_to_bin[empty_entity] = old_bin
        _mark_entity_rule_rank(state, mover_entity, rule_rank); _mark_entity_rule_rank(state, empty_entity, rule_rank)
        return
    if swap_material is None: raise ValueError('swap move requires swap_material')
    swap_entity = state.bin_to_entity[target_bin]
    state.bin_to_mat[target_bin] = mover_material; state.mat_to_bin[mover_material] = target_bin
    state.bin_to_mat[old_bin] = swap_material; state.mat_to_bin[swap_material] = old_bin
    state.bin_to_entity[target_bin] = mover_entity; state.entity_to_bin[mover_entity] = target_bin
    state.bin_to_entity[old_bin] = swap_entity; state.entity_to_bin[swap_entity] = old_bin
    _mark_entity_rule_rank(state, mover_entity, rule_rank); _mark_entity_rule_rank(state, swap_entity, rule_rank)

def _evaluate_group_window(state: State, group_materials: Sequence[str], aisle: str, window_bins: Sequence[str], *, aisle_priority: Sequence[str], desired_size: int, prefer_same_aisle_bonus: bool = True) -> Optional[dict]:
    blocked, group_in_window, empty_count, foreign_count = False, [], 0, 0
    for bin_name in window_bins:
        occupant = state.bin_to_mat.get(bin_name)
        if occupant in group_materials: group_in_window.append(occupant)
        elif occupant in (EMPTY_MARKER, None): empty_count += 1
        elif occupant in state.finalized_materials: blocked = True; break
        else: foreign_count += 1
    if blocked: return None
    group_in_aisle = sum(1 for material in group_materials if state.bin_info[state.mat_to_bin[material]]['aisle'] == aisle)
    moved_needed = sum(1 for material in group_materials if state.mat_to_bin[material] not in window_bins)
    avg_bay = float(np.mean([state.bin_info[bin_name]['bay'] for bin_name in window_bins]))
    first_bay = float(state.bin_info[window_bins[0]]['bay']) if window_bins else float('inf')
    aisle_front_rank = _front_rank_for_aisle(aisle, aisle_priority)
    shortage = max(0, desired_size - len(window_bins))
    coverage_ratio = len(group_in_window) / max(1, len(group_materials))
    prefer_hit_count, prefer_miss_count, prefer_hit_ratio = _get_group_preference_stats(group_materials, aisle, state.aisle_rules)
    selection_key = (len(group_in_window), group_in_aisle, prefer_hit_count if prefer_same_aisle_bonus else 0, -prefer_miss_count if prefer_same_aisle_bonus else 0, -shortage, -avg_bay, -aisle_front_rank, -foreign_count, -moved_needed, empty_count, -first_bay, coverage_ratio, str(aisle), str(window_bins[0]) if window_bins else '')
    return {'target_aisle': aisle, 'window_bins': list(window_bins), 'group_in_window_count': len(group_in_window), 'group_in_aisle_count': group_in_aisle, 'empty_count': empty_count, 'foreign_count': foreign_count, 'moved_needed': moved_needed, 'shortage': shortage, 'avg_bay': avg_bay, 'first_bay': first_bay, 'coverage_ratio': coverage_ratio, 'aisle_front_rank': aisle_front_rank, 'prefer_hit_count': prefer_hit_count, 'prefer_miss_count': prefer_miss_count, 'prefer_hit_ratio': prefer_hit_ratio, 'selection_key': selection_key}

def _choose_group_window(state: State, group_materials: Sequence[str], *, aisle_priority: Sequence[str], allow_cross_aisle: bool, prefer_same_aisle_bonus: bool = True) -> Optional[dict]:
    group_materials = [m for m in group_materials if m in state.mat_to_bin]
    if len(group_materials) < 2: return None
    current_aisles = {state.bin_info[state.mat_to_bin[material]]['aisle'] for material in group_materials}
    candidate_aisles = _get_common_allowed_aisles(group_materials, state)
    if not candidate_aisles: return None
    if not allow_cross_aisle:
        candidate_aisles = [aisle for aisle in candidate_aisles if aisle in current_aisles]
        if not candidate_aisles: return None
    desired_size = len(group_materials)
    best = None
    for aisle in candidate_aisles:
        aisle_bins = _sorted_bins_for_aisle(state, aisle)
        if not aisle_bins: continue
        window_size = min(desired_size, len(aisle_bins))
        for window_bins in _iter_windows(aisle_bins, window_size):
            candidate = _evaluate_group_window(state, group_materials, aisle, window_bins, aisle_priority=aisle_priority, desired_size=desired_size, prefer_same_aisle_bonus=prefer_same_aisle_bonus)
            if candidate is None: continue
            if best is None or candidate['selection_key'] > best['selection_key']: best = candidate
    return best

def _evaluate_move_candidate(state: State, mover_material: str, target_bin: str, *, move_type: str, front_priority_score: float, empty_move_penalty: float, swap_move_penalty: float, passive_swap_penalty: float, cross_aisle_move_penalty: float, center_bin: str) -> dict:
    old_bin = state.mat_to_bin[mover_material]
    old_info, new_info = state.bin_info[old_bin], state.bin_info[target_bin]
    old_distance, new_distance = _distance(state, center_bin, old_bin), _distance(state, center_bin, target_bin)
    front_shift = max(0.0, float(old_info['bay']) - float(new_info['bay']))
    cohesion_gain, front_gain = 2.0, 0.15 * (1.0 + 0.5 * float(front_priority_score)) * front_shift
    mover_gain = cohesion_gain + front_gain
    move_penalty = empty_move_penalty if move_type == 'empty' else swap_move_penalty
    if str(old_info['aisle']) != str(new_info['aisle']): move_penalty += cross_aisle_move_penalty
    if move_type == 'swap': move_penalty += passive_swap_penalty
    return {'old_distance': old_distance, 'new_distance': new_distance, 'mover_gain': mover_gain, 'swap_gain': 0.0, 'move_penalty': move_penalty, 'net_gain': mover_gain - move_penalty}

def run_reslotting_group_first(*, df_rule_analysis, df_lx03_pd3, df_bin_settings, allow_cross_aisle, lock_passive_swap, aisle_priority, aisle_rules, reserved_bin_patterns=None, association_top_k=3, association_keep_ratio=0.45, min_net_gain=0.0, empty_move_penalty=0.08, swap_move_penalty=0.45, passive_swap_penalty=0.20, cross_aisle_move_penalty=0.25, max_adjustments=None, union_priority_weight=0.70, support_priority_weight=0.20, rule_priority_weight=0.10, quality_penalty_weight=0.60, prefer_same_aisle_bonus=True):
    df_bins = df_lx03_pd3.drop_duplicates(subset=['仓位'], keep='first').copy()
    bin_info = df_bins.set_index('仓位')[['aisle', 'bay', 'level', 'sub_bay']].to_dict('index')
    bin_to_mat = df_bins.set_index('仓位')['物料'].to_dict()
    mat_to_bin = df_bins[df_bins['物料'] != EMPTY_MARKER].set_index('物料')['仓位'].to_dict()
    all_aisles = sorted(df_bins['aisle'].dropna().astype(str).unique().tolist())
    reserved_bins = _match_reserved_bins(bin_to_mat.keys(), reserved_bin_patterns)
    empty_bins = set(df_bins.loc[df_bins['物料'] == EMPTY_MARKER, '仓位']) - reserved_bins
    empty_bins_by_aisle = defaultdict(set)
    for bin_name in empty_bins: empty_bins_by_aisle[bin_info[bin_name]['aisle']].add(bin_name)
    material_weight, material_priority, _ = _build_material_stats(df_rule_analysis)
    pair_weight, neighbor_map = _build_pair_stats(df_rule_analysis, mat_to_bin)
    material_to_component, component_to_materials, _ = _build_small_cluster_maps(neighbor_map, list(mat_to_bin.keys()), top_k=association_top_k, keep_ratio=association_keep_ratio)
    entity_maps = _init_entity_maps(df_bins, df_bin_settings)
    state = State(bin_info=bin_info, bin_to_mat=bin_to_mat, mat_to_bin=mat_to_bin, empty_bins=empty_bins, empty_bins_by_aisle=dict(empty_bins_by_aisle), all_aisles=all_aisles, reserved_bins=reserved_bins, material_weight=material_weight, material_priority=material_priority, finalized_materials=set(), entity_to_bin=entity_maps['entity_to_bin'], bin_to_entity=entity_maps['bin_to_entity'], entity_label=entity_maps['entity_label'], entity_inventory=entity_maps['entity_inventory'], entity_best_rule_rank=entity_maps['entity_best_rule_rank'], aisle_rules=aisle_rules, aisle_priority=aisle_priority, protected_bin_owner={}, protected_material_owner={})
    initial_entity_to_bin = state.entity_to_bin.copy()
    df_task_rules = df_rule_analysis.copy()
    df_task_rules['rule_index'] = df_task_rules.index
    df_task_rules['group_id'] = df_task_rules['rule_materials'].apply(lambda mats: material_to_component.get(next((m for m in mats if m in material_to_component), None)))
    group_rows = []
    for group_id, group_materials in component_to_materials.items():
        active_materials = [material for material in group_materials if material in state.mat_to_bin and state.mat_to_bin.get(material) not in state.reserved_bins]
        if len(active_materials) < 2: continue
        group_rules = df_task_rules[df_task_rules['group_id'] == group_id]
        rep_material = _representative_material(state, active_materials)
        support_rank = float(group_rules['support_rank'].max()) if not group_rules.empty else 0.0
        raw_support = float(group_rules['支持度'].max()) if ('支持度' in group_rules.columns and not group_rules.empty) else 0.0
        rule_weight = float(group_rules['rule_weight'].max()) if not group_rules.empty else 0.0
        evidence_count = int(group_rules['evidence_count'].sum()) if not group_rules.empty else 0
        group_union_support = float(group_rules['group_union_support'].max()) if ('group_union_support' in group_rules.columns and not group_rules.empty and group_rules['group_union_support'].notna().any()) else raw_support
        group_union_order_count = float(group_rules['group_union_order_count'].max()) if ('group_union_order_count' in group_rules.columns and not group_rules.empty and group_rules['group_union_order_count'].notna().any()) else np.nan
        group_rows.append({'group_id': group_id, 'rule_index': int(group_rules['rule_index'].min()) if not group_rules.empty else None, 'support_rank': support_rank, 'raw_support': raw_support, 'group_union_support': group_union_support, 'group_union_order_count': group_union_order_count, 'rule_weight': rule_weight, 'evidence_count': evidence_count, 'rule_materials': active_materials, 'competition_materials': active_materials, 'anchor_material': rep_material})
    df_group_tasks = pd.DataFrame(group_rows)
    if not df_group_tasks.empty:
        df_group_tasks['union_support_rank'] = _safe_rank(df_group_tasks['group_union_support'].fillna(df_group_tasks['raw_support']))
        df_group_tasks['rule_weight_rank'] = _safe_rank(df_group_tasks['rule_weight'])
        df_group_tasks['front_priority_score'] = union_priority_weight * df_group_tasks['union_support_rank'] + support_priority_weight * df_group_tasks['support_rank'] + rule_priority_weight * df_group_tasks['rule_weight_rank']
    if df_group_tasks.empty:
        empty_adjustments = pd.DataFrame(columns=['关联规则索引', '关联组', '规则物料', '竞争组物料', '锚点物料', '锚点库位', '目标通道', '移动物料', '目标库位物料', '原库位', '目标库位', '移动类型', '交换物料', '调整前距离', '调整后距离', '移动物料收益', '交换物料收益', '执行惩罚', '估计净收益', '前端优先级', '关联组并集频率', '关联组并集订单数', '目标窗口平均bay'])
        return {'adjustments': empty_adjustments, 'skips': pd.DataFrame(columns=['跳过层级', '关联规则索引', '关联组', '规则物料', '竞争组物料', '锚点物料', '目标通道', '涉及物料', '待处理物料', '原因代码', '原因说明']), 'move_list': pd.DataFrame(columns=['调仓组', '物料', '原库位', '新库位', '总库存量']), 'rule_tasks': df_group_tasks, 'material_weight': material_weight, 'pair_weight': pair_weight, 'neighbor_map': neighbor_map, 'reserved_bins': sorted(reserved_bins)}
    df_group_tasks = df_group_tasks.sort_values(['front_priority_score', 'group_union_support', 'support_rank', 'rule_weight', 'evidence_count', 'group_id'], ascending=[False, False, False, False, False, True]).reset_index(drop=True)
    df_group_tasks['priority_rank'] = np.arange(1, len(df_group_tasks) + 1, dtype=int)
    if not df_group_tasks.empty:
        df_group_tasks['preferred_aisles'] = df_group_tasks['rule_materials'].apply(lambda mats: sorted({a for m in mats for a in _get_material_aisle_policy(m, aisle_rules).get('prefer_aisles', [])}))
        df_group_tasks['forbidden_aisles'] = df_group_tasks['rule_materials'].apply(lambda mats: sorted({a for m in mats for a in _get_material_aisle_policy(m, aisle_rules).get('forbid_aisles', [])}))
    adjustments, skips = [], []
    for _, task in df_group_tasks.iterrows():
        if max_adjustments is not None and len(adjustments) >= max_adjustments:
            _append_skip(skips, 'rule', task['rule_index'], task['group_id'], task['rule_materials'], task['competition_materials'], task['anchor_material'], None, task['rule_materials'], [], 'budget_exhausted')
            continue
        group_id, rule_index, rule_rank = task['group_id'], task['rule_index'], task['priority_rank']
        front_priority_score = float(task['front_priority_score'])
        group_materials = [m for m in task['rule_materials'] if m in state.mat_to_bin]
        if len(group_materials) < 2:
            _append_skip(skips, 'rule', rule_index, group_id, task['rule_materials'], task['competition_materials'], task['anchor_material'], None, group_materials, [], 'insufficient_active_materials')
            continue
        common_allowed_aisles = _get_common_allowed_aisles(group_materials, state)
        if not common_allowed_aisles:
            _append_skip(skips, 'rule', rule_index, group_id, task['rule_materials'], task['competition_materials'], task['anchor_material'], None, group_materials, [], 'no_common_allowed_aisle')
            continue
        choice = _choose_group_window(state, group_materials, aisle_priority=aisle_priority, allow_cross_aisle=allow_cross_aisle, prefer_same_aisle_bonus=prefer_same_aisle_bonus)
        if choice is None:
            _append_skip(skips, 'rule', rule_index, group_id, task['rule_materials'], task['competition_materials'], task['anchor_material'], None, group_materials, [], 'no_target_window')
            continue
        target_aisle, window_bins = choice['target_aisle'], choice['window_bins']
        center_bin = window_bins[len(window_bins) // 2]
        current_group_bins = [state.mat_to_bin[m] for m in group_materials if m in state.mat_to_bin]
        if current_group_bins and all(bin_name in set(window_bins) for bin_name in current_group_bins):
            state.finalized_materials.update(group_materials)
            _append_skip(skips, 'rule', rule_index, group_id, task['rule_materials'], task['competition_materials'], task['anchor_material'], target_aisle, group_materials, [], 'already_grouped')
            continue
        occupant_map = {bin_name: state.bin_to_mat.get(bin_name) for bin_name in window_bins}
        target_bins_to_fill, candidate_materials = _build_group_assignment_plan(state, group_materials, window_bins, target_aisle, occupant_map, quality_penalty_weight=quality_penalty_weight)
        moved_any = False
        pending_assignments = list(zip(target_bins_to_fill, candidate_materials))
        for target_bin, mover_material in pending_assignments:
            if max_adjustments is not None and len(adjustments) >= max_adjustments: break
            occupant = state.bin_to_mat.get(target_bin)
            if occupant not in (EMPTY_MARKER, None) and occupant in state.finalized_materials and occupant not in group_materials: continue
            old_bin = state.mat_to_bin.get(mover_material)
            if old_bin is None:
                _append_skip(skips, 'material', rule_index, group_id, group_materials, group_materials, task['anchor_material'], target_aisle, mover_material, candidate_materials, 'missing_bin')
                continue
            if target_bin == old_bin: state.finalized_materials.add(mover_material); continue
            if not _is_aisle_allowed_for_material(mover_material, target_aisle, state.aisle_rules):
                _append_skip(skips, 'material', rule_index, group_id, group_materials, group_materials, task['anchor_material'], target_aisle, mover_material, candidate_materials, 'target_aisle_forbidden')
                continue
            old_info = state.bin_info[old_bin]
            move_type = 'empty' if occupant in (EMPTY_MARKER, None) else 'swap'
            swap_material = occupant if move_type == 'swap' else None
            if swap_material is not None and (swap_material in group_materials or swap_material in state.finalized_materials): continue
            if swap_material is not None and not _is_aisle_allowed_for_material(swap_material, old_info['aisle'], state.aisle_rules):
                _append_skip(skips, 'material', rule_index, group_id, group_materials, group_materials, task['anchor_material'], target_aisle, swap_material, candidate_materials, 'swap_target_aisle_forbidden')
                continue
            move_eval = _evaluate_move_candidate(state, mover_material, target_bin, move_type=move_type, front_priority_score=front_priority_score, empty_move_penalty=empty_move_penalty, swap_move_penalty=swap_move_penalty, passive_swap_penalty=passive_swap_penalty, cross_aisle_move_penalty=cross_aisle_move_penalty, center_bin=center_bin)
            if move_eval['net_gain'] <= min_net_gain:
                _append_skip(skips, 'material', rule_index, group_id, group_materials, group_materials, task['anchor_material'], target_aisle, mover_material, candidate_materials, 'no_better_candidate')
                continue
            _apply_material_move(state, mover_material=mover_material, target_bin=target_bin, move_type=move_type, rule_rank=rule_rank, swap_material=swap_material)
            moved_any = True
            state.finalized_materials.add(mover_material)
            if move_type == 'swap' and swap_material is not None and lock_passive_swap: state.finalized_materials.add(swap_material)
            adjustments.append({'rule_index': rule_index, 'group_id': group_id, 'rule_materials': ', '.join(group_materials), 'competition_materials': ', '.join(group_materials), 'anchor_material': task['anchor_material'], 'anchor_bin': state.mat_to_bin.get(task['anchor_material']) if task['anchor_material'] in state.mat_to_bin else None, 'target_aisle': target_aisle, 'material_mover': mover_material, 'move_type': move_type, 'from_bin': old_bin, 'to_bin': target_bin, 'swap_material': swap_material, 'old_distance': move_eval['old_distance'], 'new_distance': move_eval['new_distance'], 'mover_gain': move_eval['mover_gain'], 'swap_gain': move_eval['swap_gain'], 'move_penalty': move_eval['move_penalty'], 'net_gain': move_eval['net_gain'], 'front_priority_score': front_priority_score, 'group_union_support': task.get('group_union_support'), 'group_union_order_count': task.get('group_union_order_count'), 'window_avg_bay': choice['avg_bay'], 'window_prefer_hit_count': choice.get('prefer_hit_count'), 'window_prefer_miss_count': choice.get('prefer_miss_count'), 'window_prefer_hit_ratio': choice.get('prefer_hit_ratio')})
        if not moved_any: _append_skip(skips, 'rule', rule_index, group_id, group_materials, group_materials, task['anchor_material'], target_aisle, group_materials, candidate_materials, 'already_grouped' if not target_bins_to_fill else 'no_candidate')
        state.finalized_materials.update(group_materials)
    df_adjustments = pd.DataFrame(adjustments)
    if not df_adjustments.empty: df_adjustments = df_adjustments.merge(df_bin_settings, left_on='to_bin', right_index=True, how='left')
    df_skips = pd.DataFrame(skips)
    df_move_list = _build_move_list(state, initial_entity_to_bin)
    df_adjustments, df_skips, df_move_list, df_group_tasks = _format_outputs(df_adjustments, df_skips, df_move_list, df_group_tasks)
    if not df_group_tasks.empty:
        df_group_tasks['rule_materials'] = df_group_tasks['rule_materials'].apply(list)
        df_group_tasks['competition_materials'] = df_group_tasks['competition_materials'].apply(list)
    return {'adjustments': df_adjustments, 'skips': df_skips, 'move_list': df_move_list, 'rule_tasks': df_group_tasks, 'material_weight': material_weight, 'pair_weight': pair_weight, 'neighbor_map': neighbor_map, 'reserved_bins': sorted(reserved_bins)}

# ==========================================
# 6. Services (Pipeline Orchestration)
# ==========================================
ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]

class _StreamingWriter(io.TextIOBase):
    def __init__(self, callback: LogFn | None = None):
        self._buffer = io.StringIO()
        self._callback = callback
    @property
    def encoding(self) -> str: return "utf-8"
    def write(self, text: str) -> int:
        value = str(text)
        self._buffer.write(value)
        if self._callback is not None and value: self._callback(value)
        return len(value)
    def flush(self) -> None: return None
    def getvalue(self) -> str: return self._buffer.getvalue()

def _capture_stdout(func, *args, log_callback: LogFn | None = None, **kwargs):
    writer = _StreamingWriter(log_callback)
    with contextlib.redirect_stdout(writer):
        result = func(*args, **kwargs)
    return result, writer.getvalue()

def _emit_log(log_callback: LogFn | None, text: str) -> None:
    if log_callback is not None: log_callback(text if text.endswith("\n") else text + "\n")

def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(f"文件不存在: {p}")
    suffix = p.suffix.lower()
    if suffix in {".xlsx", ".xls"}: return pd.read_excel(p)
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try: return pd.read_csv(p, encoding=enc)
            except UnicodeDecodeError: continue
        raise ValueError(f"无法读取 CSV 文件: {p}")
    raise ValueError(f"不支持的文件类型: {p.suffix}")

EXCLUDED_BINS = {"31-01-1A", "51-01-1A"}

def load_ship_df(config: dict[str, Any], path_override: str | Path | None = None) -> pd.DataFrame:
    path = resolve_ship_data_path(config, path_override)
    if path is None: raise ValueError("尚未上传历史发货数据。请在 01 配置与数据源中上传文件。")
    df = read_table(path)
    cols = set(df.columns)
    if {"TO number", "Material"}.issubset(cols):
        if {"MTy", "Source bin", "Dest.bin"}.issubset(cols):
            df = df[(df["MTy"] == 981) & (~df["Source bin"].isin(EXCLUDED_BINS)) & (~df["Dest.bin"].isin(EXCLUDED_BINS))]
        return df[["TO number", "Material"]].copy()
    raise ValueError(f"发货数据 {path.name} 至少需要包含列: TO number, Material")

def load_lx03_pd3(config: dict[str, Any], path_override: str | Path | None = None) -> pd.DataFrame:
    path = resolve_lx03_path(config, path_override)
    if path is None: raise ValueError("尚未上传 LX03 库位数据。请在 01 配置与数据源中上传文件。")
    df = read_table(path)
    if "仓储类型" in df.columns: df = df[df["仓储类型"] == "PD3"].copy()
    required = {"物料", "仓位"}
    if not required.issubset(df.columns): raise ValueError(f"LX03 文件 {path.name} 至少需要包含列: 物料, 仓位")
    if "总库存量" not in df.columns: df["总库存量"] = pd.NA
    df = df[["物料", "仓位", "总库存量"]].copy()
    df["aisle"] = df["仓位"].astype(str).str[:2]
    df["bay"] = pd.to_numeric(df["仓位"].astype(str).str[3:5], errors="coerce").fillna(0).astype(int)
    df["level"] = df["仓位"].astype(str).str[6:7]
    df["sub_bay"] = df["仓位"].astype(str).str[7:8]
    return df

def load_bin_settings(config: dict[str, Any], df_lx03_pd3: pd.DataFrame, path_override: str | Path | None = None) -> pd.DataFrame:
    path = resolve_bin_settings_path(config, path_override)
    if path and path.exists():
        df = read_table(path)
        if {"物料", "仓位"}.issubset(df.columns):
            return df[["物料", "仓位"]].dropna().rename(columns={"物料": "to_bin_material", "仓位": "bin"}).set_index("bin")
    return df_lx03_pd3[["物料", "仓位"]].dropna().rename(columns={"物料": "to_bin_material", "仓位": "bin"}).set_index("bin")

def preview_data_sources(config: dict[str, Any], *, ship_path: str | Path | None = None, lx03_path: str | Path | None = None, bin_settings_path: str | Path | None = None, limit: int = 200) -> dict[str, Any]:
    ship = load_ship_df(config, ship_path)
    lx03 = load_lx03_pd3(config, lx03_path)
    bins = load_bin_settings(config, lx03, bin_settings_path).reset_index()
    n = max(1, min(int(limit), 500))
    return {"ship": ship.head(n).fillna(""), "lx03": lx03.head(n).fillna(""), "bin_settings": bins.head(n).fillna(""), "counts": {"ship": int(len(ship)), "lx03": int(len(lx03)), "bin_settings": int(len(bins))}}

def tidy_rules_output(rules_fpgrowth: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["antecedents", "consequents", "support", "confidence", "lift", "group_union_order_count", "group_union_support"]
    available = [c for c in base_cols if c in rules_fpgrowth.columns]
    rules = rules_fpgrowth[available].copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
    rules["len_antecedents"] = rules["antecedents"].apply(lambda s: 0 if not s else len(str(s).split(", ")))
    rules["len_consequents"] = rules["consequents"].apply(lambda s: 0 if not s else len(str(s).split(", ")))
    rule_single_mask = (rules["len_antecedents"] == 1) & (rules["len_consequents"] == 1)
    def is_new_wh(mat_bin: str) -> str:
        try: return "新仓" if int(str(mat_bin).split(" @ ")[1][:2]) <= 7 else "老仓"
        except: return "未知"
    rules["ant_mats"], rules["cons_mats"], rules["same_wh"], rules["same_aisle"] = pd.NA, pd.NA, pd.NA, pd.NA
    rules.loc[rule_single_mask, "ant_mats"] = rules.loc[rule_single_mask, "antecedents"].apply(is_new_wh)
    rules.loc[rule_single_mask, "cons_mats"] = rules.loc[rule_single_mask, "consequents"].apply(is_new_wh)
    rules.loc[rule_single_mask, "same_wh"] = (rules.loc[rule_single_mask, "ant_mats"] == rules.loc[rule_single_mask, "cons_mats"])
    rules.loc[rule_single_mask, "same_aisle"] = (rules.loc[rule_single_mask, "antecedents"].apply(lambda x: str(x).split(" @ ")[1][:2]) == rules.loc[rule_single_mask, "consequents"].apply(lambda x: str(x).split(" @ ")[1][:2]))
    rules = rules.rename(columns={"antecedents": "前项物料", "consequents": "后项物料", "support": "支持度", "confidence": "置信度", "lift": "提升度", "ant_mats": "老仓", "cons_mats": "新仓", "same_wh": "同仓库拣货", "same_aisle": "同通道拣货", "group_union_order_count": "关联组并集订单数", "group_union_support": "关联组并集频率"})
    sort_cols = [c for c in ["关联组并集频率", "支持度", "置信度", "提升度"] if c in rules.columns]
    if sort_cols: rules = rules.sort_values(by=sort_cols, ascending=False)
    return rules

def run_fpgrowth_pipeline(config: dict[str, Any], progress: ProgressFn | None = None, *, ship_path: str | Path | None = None, lx03_path: str | Path | None = None, log_callback: LogFn | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir()
    if progress: progress(0.10, "读取发货数据")
    _emit_log(log_callback, "[准备] 读取发货数据…")
    ship_df = load_ship_df(config, ship_path)
    _emit_log(log_callback, f"[准备] 发货数据读取完成：{len(ship_df)} 行")
    if progress: progress(0.24, "读取 LX03 并筛选 PD3")
    _emit_log(log_callback, "[准备] 读取 LX03 并筛选 PD3…")
    df_lx03_pd3 = load_lx03_pd3(config, lx03_path)
    _emit_log(log_callback, f"[准备] PD3 库位读取完成：{len(df_lx03_pd3)} 行")
    ship_materials = ship_df["Material"].astype(str).str.strip()
    lx03_materials = set(df_lx03_pd3["物料"].astype(str).str.strip())
    matched_mask = ship_materials.isin(lx03_materials)
    matched_rows, matched_unique = int(matched_mask.sum()), int(ship_materials[matched_mask].nunique())
    _emit_log(log_callback, f"[校验] 发货数据与 PD3 匹配：{matched_rows}/{len(ship_df)} 行，{matched_unique} 个唯一物料")
    if matched_rows == 0: raise ValueError("发货数据与当前 LX03 的 PD3 物料没有交集。请确认历史发货数据和 LX03 属于同一仓库/期间。")
    lx03_for_rules = df_lx03_pd3[["物料", "仓位"]].copy()
    if progress: progress(0.38, "执行 FP-Growth 关联分析")
    rules_fpgrowth, logs = _capture_stdout(run_fpgrowth, ship_df, lx03_for_rules, min_support=float(config.get("min_support", 0.0003)), log_callback=log_callback)
    if rules_fpgrowth is None or rules_fpgrowth.empty:
        if progress: progress(1.0, "未找到满足阈值的关联规则")
        return {"ok": False, "message": "未找到频繁项集或关联规则。", "logs": logs}
    if progress: progress(0.82, "整理并保存关联规则")
    _emit_log(log_callback, "[整理] 关联规则计算完成，正在整理字段并保存…")
    rules_tidy = tidy_rules_output(rules_fpgrowth)
    rules_path = output_dir / RULES_CSV_NAME
    rules_tidy.to_csv(rules_path, index=False, encoding="utf-8-sig")
    clear_cache()
    if progress: progress(1.0, "关联分析完成")
    _emit_log(log_callback, f"[完成] 已保存 {len(rules_tidy)} 条关联规则：{rules_path.name}")
    return {"ok": True, "message": "关联分析运行完成。", "logs": logs, "rules_csv": str(rules_path), "rule_count": int(len(rules_tidy)), "preview": rules_tidy.head(200).fillna("")}

def _all_group_material_map(df_rule_tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not df_rule_tasks.empty:
        for _, row in df_rule_tasks.iterrows():
            materials = row.get("rule_materials", [])
            if not isinstance(materials, list): continue
            for material in materials:
                rows.append({"物料": material, "关联组": row.get("group_id"), "优先级序号": row.get("priority_rank"), "前端优先级": row.get("front_priority_score"), "关联组并集频率": row.get("group_union_support"), "关联组并集订单数": row.get("group_union_order_count"), "锚点物料": row.get("anchor_material")})
    columns = ["物料", "关联组", "优先级序号", "前端优先级", "关联组并集频率", "关联组并集订单数", "锚点物料"]
    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["物料"], keep="first") if rows else pd.DataFrame(columns=columns)

def run_reslotting_pipeline(config: dict[str, Any], progress: ProgressFn | None = None, *, lx03_path: str | Path | None = None, bin_settings_path: str | Path | None = None) -> dict[str, Any]:
    output_dir = ensure_output_dir()
    rules_path = output_dir / RULES_CSV_NAME
    if not rules_path.exists(): raise FileNotFoundError(f"未找到关联分析结果文件: {rules_path.name}，请先执行关联分析。")
    if progress: progress(0.08, "读取关联规则")
    df_related_mat = read_table(rules_path)
    if progress: progress(0.18, "读取当前 PD3 布局")
    df_lx03_pd3 = load_lx03_pd3(config, lx03_path)
    df_bin_settings = load_bin_settings(config, df_lx03_pd3, bin_settings_path)
    if progress: progress(0.28, "构建规则与物料关联网络")
    df_rule_analysis, _df_pair_analysis = prepare_rule_analysis(df_related_mat)
    if progress: progress(0.42, "执行关联组优先调仓")
    result, logs = _capture_stdout(run_reslotting_group_first, df_rule_analysis=df_rule_analysis, df_lx03_pd3=df_lx03_pd3, df_bin_settings=df_bin_settings, allow_cross_aisle=ALLOW_CROSS_AISLE, lock_passive_swap=LOCK_PASSIVE_SWAP, aisle_priority=config["business_rules"].get("aisle_priority", []), aisle_rules=config["business_rules"].get("aisle_rules", []), reserved_bin_patterns=config["business_rules"].get("reserved_bin_patterns", []), max_adjustments=MAX_ADJUSTMENTS)
    df_adjustments, df_skips, df_move_list, df_rule_tasks = result["adjustments"], result["skips"], result["move_list"], result["rule_tasks"]
    df_group_material_map = _all_group_material_map(df_rule_tasks)
    if progress: progress(0.70, "生成调仓前后布局")
    move_replace = df_move_list[["原库位", "新库位"]].set_index("原库位")["新库位"] if not df_move_list.empty else pd.Series(dtype=object)
    df_pd3_original = df_lx03_pd3.copy().merge(df_group_material_map, on="物料", how="left")
    df_pd3_adjusted = df_pd3_original.copy()
    if not move_replace.empty:
        df_pd3_adjusted["仓位"] = df_pd3_adjusted["仓位"].replace(move_replace.to_dict())
        df_pd3_adjusted["aisle"] = df_pd3_adjusted["仓位"].astype(str).str[:2]
        df_pd3_adjusted["bay"] = pd.to_numeric(df_pd3_adjusted["仓位"].astype(str).str[3:5], errors="coerce").fillna(0).astype(int)
        df_pd3_adjusted["level"] = df_pd3_adjusted["仓位"].astype(str).str[6:7]
        df_pd3_adjusted["sub_bay"] = df_pd3_adjusted["仓位"].astype(str).str[7:8]
    before_path, after_path = output_dir / BEFORE_LAYOUT_XLSX_NAME, output_dir / AFTER_LAYOUT_XLSX_NAME
    df_pd3_original.to_excel(before_path, index=False)
    df_pd3_adjusted.to_excel(after_path, index=False)
    results_path = output_dir / RESULTS_XLSX_NAME
    with pd.ExcelWriter(results_path) as writer:
        df_adjustments.to_excel(writer, sheet_name="adjustments", index=False)
        df_skips.to_excel(writer, sheet_name="skips", index=False)
        df_move_list.to_excel(writer, sheet_name="move_list", index=False)
        df_rule_tasks.to_excel(writer, sheet_name="group_tasks", index=False)
        df_group_material_map.to_excel(writer, sheet_name="group_material_map", index=False)
    if progress: progress(0.86, "生成 SVG 矢量布局图")
    moved_materials = set(df_move_list["物料"].astype(str)) if not df_move_list.empty and "物料" in df_move_list.columns else set()
    before_svg, after_svg = output_dir / BEFORE_LAYOUT_SVG_NAME, output_dir / AFTER_LAYOUT_SVG_NAME
    render_layout_pair(df_pd3_original, df_pd3_adjusted, before_svg, after_svg, moved_materials=moved_materials)
    before_plot, after_plot = build_layout_plot_pair(df_pd3_original, df_pd3_adjusted, moved_materials=moved_materials)
    clear_cache()
    if progress: progress(1.0, "调仓方案与可视化已生成")
    return {"ok": True, "message": "调仓运行完成。", "logs": logs, "summary": {"identified_groups": int(df_rule_tasks["group_id"].nunique()) if not df_rule_tasks.empty and "group_id" in df_rule_tasks.columns else 0, "adjustment_rows": int(len(df_adjustments)), "move_rows": int(len(df_move_list)), "skip_rows": int(len(df_skips))}, "files": [str(p) for p in [results_path, before_path, after_path, before_svg, after_svg] if p.exists()], "plots": {"before": before_plot, "after": after_plot}, "previews": {"adjustments": df_adjustments.head(200).fillna(""), "move_list": df_move_list.head(200).fillna(""), "skips": df_skips.head(200).fillna(""), "group_tasks": df_rule_tasks.head(200).fillna(""), "group_material_map": df_group_material_map.head(200).fillna("")}}

def get_latest_summary() -> dict[str, Any]:
    output_dir = ensure_output_dir()
    results_path, rules_path = output_dir / RESULTS_XLSX_NAME, output_dir / RULES_CSV_NAME
    summary = {"rule_count": 0, "identified_groups": 0, "adjustment_rows": 0, "move_rows": 0, "skip_rows": 0}
    if rules_path.exists():
        try: summary["rule_count"] = int(len(read_table(rules_path)))
        except Exception: pass
    if results_path.exists():
        for sheet, key in [("adjustments", "adjustment_rows"), ("move_list", "move_rows"), ("skips", "skip_rows")]:
            try: summary[key] = int(len(pd.read_excel(results_path, sheet_name=sheet)))
            except Exception: pass
        try:
            tasks = pd.read_excel(results_path, sheet_name="group_tasks")
            if "group_id" in tasks.columns: summary["identified_groups"] = int(tasks["group_id"].nunique())
        except Exception: pass
    return summary

def get_output_files() -> list[str]:
    output_dir = ensure_output_dir()
    names = [RULES_CSV_NAME, RESULTS_XLSX_NAME, BEFORE_LAYOUT_XLSX_NAME, AFTER_LAYOUT_XLSX_NAME, BEFORE_LAYOUT_SVG_NAME, AFTER_LAYOUT_SVG_NAME]
    return [str(output_dir / name) for name in names if (output_dir / name).exists()]

def get_layout_plots():
    output_dir = ensure_output_dir()
    before_path, after_path = output_dir / BEFORE_LAYOUT_XLSX_NAME, output_dir / AFTER_LAYOUT_XLSX_NAME
    if not before_path.exists() or not after_path.exists(): return empty_layout_plot(), empty_layout_plot()
    try:
        before_df, after_df = pd.read_excel(before_path), pd.read_excel(after_path)
        moved_materials: set[str] = set()
        results_path = output_dir / RESULTS_XLSX_NAME
        if results_path.exists():
            try:
                move_df = pd.read_excel(results_path, sheet_name="move_list")
                if "物料" in move_df.columns: moved_materials = set(move_df["物料"].dropna().astype(str))
            except Exception: pass
        return build_layout_plot_pair(before_df, after_df, moved_materials=moved_materials)
    except Exception as exc:
        message = f"布局结果读取失败：{exc}"
        return empty_layout_plot(message), empty_layout_plot(message)

def get_layout_html() -> tuple[str, str]:
    output_dir = ensure_output_dir()
    return svg_as_html(output_dir / BEFORE_LAYOUT_SVG_NAME), svg_as_html(output_dir / AFTER_LAYOUT_SVG_NAME)

# ==========================================
# 7. Gradio UI (main.py)
# ==========================================
APP_TITLE = "关联调仓解决方案"
PREVIEW_ROWS = 200
PAGE_SIZE_CHOICES = [50, 100, 200, 500]
RULE_COLUMNS = ["规则名称", "匹配方式", "物料模式", "禁止通道", "优先通道"]

def _split_tokens(value: Any) -> list[str]:
    if value is None: return []
    if isinstance(value, (list, tuple, set)): return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).replace("，", ",").replace("\n", ",")
    return [x.strip() for x in text.split(",") if x.strip()]

def _rules_frame(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for rule in config.get("business_rules", {}).get("aisle_rules", []):
        rows.append({"规则名称": rule.get("name", ""), "匹配方式": rule.get("match_mode", "prefix"), "物料模式": rule.get("pattern", ""), "禁止通道": ", ".join(rule.get("forbid_aisles", [])), "优先通道": ", ".join(rule.get("prefer_aisles", []))})
    return pd.DataFrame(rows, columns=RULE_COLUMNS)

def _rules_from_frame(frame: Any) -> list[dict[str, Any]]:
    if frame is None: return []
    if not isinstance(frame, pd.DataFrame): frame = pd.DataFrame(frame, columns=RULE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in frame.fillna("").iterrows():
        pattern = str(row.get("物料模式", "")).strip()
        if not pattern: continue
        rows.append({"name": str(row.get("规则名称", "")).strip(), "match_mode": str(row.get("匹配方式", "prefix")).strip() or "prefix", "pattern": pattern, "forbid_aisles": _split_tokens(row.get("禁止通道", "")), "prefer_aisles": _split_tokens(row.get("优先通道", ""))})
    return rows

def _uploaded_path(value: Any) -> str | None:
    if value is None: return None
    if isinstance(value, (list, tuple)): return _uploaded_path(value[0]) if value else None
    if isinstance(value, (str, Path)): return str(value)
    if isinstance(value, dict):
        for key in ("path", "name"):
            if value.get(key): return str(value[key])
    name = getattr(value, "name", None)
    return str(name) if name else None

def _selected_upload(uploaded: Any) -> str:
    return _uploaded_path(uploaded) or ""

def _build_config(whn: str, plant: str, start_date: str, end_date: str, min_support: float, aisle_priority: str, reserved_bins: str, rules: Any) -> dict[str, Any]:
    return {"whn": whn, "plant": plant, "start_date": start_date, "end_date": end_date, "min_support": min_support, "data_sources": {"ship_data_path": "", "lx03_path": "", "bin_settings_path": ""}, "business_rules": {"aisle_priority": _split_tokens(aisle_priority), "reserved_bin_patterns": _split_tokens(reserved_bins), "aisle_rules": _rules_from_frame(rules)}}

def _source_status_markdown(status: dict[str, Any]) -> str:
    labels = {"ship": "历史发货数据", "lx03": "LX03 库位数据", "bin_settings": "库位设定"}
    lines = ["### 数据源状态"]
    for key in ("ship", "lx03", "bin_settings"):
        item = status[key]
        mark = "✅" if item["exists"] else "❌"
        lines.append(f"- {mark} {labels[key]}：{item['path']}")
    return "\n".join(lines)

def _summary_markdown(summary: dict[str, Any] | None = None) -> str:
    s = summary or get_latest_summary()
    return f"### 当前结果摘要\n关联规则：{s.get('rule_count', 0)}  |  关联组：{s.get('identified_groups', 0)}  |  调仓建议：{s.get('adjustment_rows', 0)}  |  实际移动：{s.get('move_rows', 0)}  |  跳过记录：{s.get('skip_rows', 0)}"

def _combined_log(title: str, result: dict[str, Any]) -> str:
    chunks = [f"[{title}]", str(result.get("message", "") or "")]
    logs = str(result.get("logs", "") or "").strip()
    if logs: chunks += ["\n", logs]
    return "\n".join(chunks).strip()

def _page_result(table_name: str, page: int, page_size: int, search: str):
    try:
        df, page, pages, total = get_page(table_name, int(page or 1), int(page_size or PREVIEW_ROWS), search or "")
        status = f"第 {page} / {pages} 页 · 共 {total} 行 · 当前发送 {len(df)} 行到浏览器"
        return df, page, status
    except Exception as exc:
        return pd.DataFrame(), 1, f"读取失败：{exc}"

def prev_page(table_name, page, page_size, search): return _page_result(table_name, max(1, int(page or 1) - 1), page_size, search)
def next_page(table_name, page, page_size, search): return _page_result(table_name, int(page or 1) + 1, page_size, search)

def add_rule_row(frame):
    if not isinstance(frame, pd.DataFrame): frame = pd.DataFrame(frame or [], columns=RULE_COLUMNS)
    new = pd.DataFrame([{"规则名称": "", "匹配方式": "prefix", "物料模式": "", "禁止通道": "", "优先通道": ""}])
    return pd.concat([frame, new], ignore_index=True)

def load_config_ui():
    cfg = load_config()
    br = cfg.get("business_rules", {})
    return (cfg.get("whn", ""), cfg.get("plant", ""), cfg.get("start_date", ""), cfg.get("end_date", ""), float(cfg.get("min_support", 0.0003)), ", ".join(br.get("aisle_priority", [])), "\n".join(br.get("reserved_bin_patterns", [])), _rules_frame(cfg), "配置已从 config.json 重新载入。数据文件仍使用当前页面已上传的文件。")

def save_config_ui(whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules):
    cfg = _build_config(whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules)
    saved = save_config(cfg)
    return _rules_frame(saved), "业务配置已保存到 config.json。数据文件不会写入配置。"

def check_sources_ui(ship_upload, lx03_upload, bin_upload):
    def item(uploaded: Any, *, optional: bool = False) -> dict[str, Any]:
        path_text = _selected_upload(uploaded)
        path = Path(path_text) if path_text else None
        if path is not None: return {"path": path.name, "exists": path.exists()}
        if optional: return {"path": "未上传，将根据 LX03 自动生成", "exists": True}
        return {"path": "未上传", "exists": False}
    return _source_status_markdown({"ship": item(ship_upload), "lx03": item(lx03_upload), "bin_settings": item(bin_upload, optional=True)})

def preview_sources_ui(ship_upload, lx03_upload, bin_upload):
    cfg = load_config()
    try:
        runtime_ship, runtime_lx03, runtime_bins = _selected_upload(ship_upload), _selected_upload(lx03_upload), _selected_upload(bin_upload)
        if not runtime_ship: raise ValueError("请先上传历史发货数据。")
        if not runtime_lx03: raise ValueError("请先上传 LX03 库位数据。")
        result = preview_data_sources(cfg, ship_path=runtime_ship, lx03_path=runtime_lx03, bin_settings_path=runtime_bins, limit=PREVIEW_ROWS)
        counts = result["counts"]
        status = f"✅ 数据读取成功。发货数据 {counts['ship']} 行，PD3 库位 {counts['lx03']} 行，库位设定 {counts['bin_settings']} 行；下方各表最多预览前 {PREVIEW_ROWS} 行。"
        return status, result["ship"], result["lx03"], result["bin_settings"]
    except Exception as exc:
        return f"❌ 数据读取失败：{exc}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def _trim_stream_log(text: str, limit: int = 50_000) -> str:
    if len(text) <= limit: return text
    return "[较早日志已省略；完整计算结果仍写入 output/]\n…\n" + text[-limit:]

def stream_fpgrowth_log(whn, plant, start_date, end_date, min_support, ship_upload, lx03_upload, bin_upload, aisle_priority, reserved_bins, rules, progress=gr.Progress()):
    cfg = _build_config(whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules)
    runtime_ship, runtime_lx03 = _selected_upload(ship_upload), _selected_upload(lx03_upload)
    if not runtime_ship or not runtime_lx03:
        missing = []
        if not runtime_ship: missing.append("历史发货数据")
        if not runtime_lx03: missing.append("LX03 库位数据")
        message = "请先在 01 配置与数据源上传：" + "、".join(missing)
        yield f"[关联分析]\nERROR: {message}\n", {"ok": False, "error": message, "message": "关联分析失败"}
        return
    events: queue.Queue[tuple] = queue.Queue()
    holder: dict[str, Any] = {}
    def on_progress(value: float, desc: str) -> None: events.put(("progress", float(value), str(desc)))
    def on_log(text: str) -> None:
        if text: events.put(("log", str(text)))
    def worker() -> None:
        try: holder["result"] = run_fpgrowth_pipeline(cfg, progress=on_progress, ship_path=runtime_ship, lx03_path=runtime_lx03, log_callback=on_log)
        except Exception as exc: holder["error"] = exc
        finally: events.put(("done",))
    thread = threading.Thread(target=worker, daemon=True, name="fpgrowth-ui-worker")
    progress(0, desc="准备关联分析")
    thread.start()
    streamed_log = "[关联分析]\n"
    yield streamed_log, gr.skip()
    done = False
    while not done:
        event = events.get()
        batch = [event]
        while True:
            try: batch.append(events.get_nowait())
            except queue.Empty: break
        changed = False
        for item in batch:
            kind = item[0]
            if kind == "log": streamed_log += item[1]; changed = True
            elif kind == "progress": progress(item[1], desc=item[2])
            elif kind == "done": done = True
        if changed: yield _trim_stream_log(streamed_log), gr.skip()
    if "error" in holder:
        progress(1.0, desc="关联分析失败")
        exc = holder["error"]
        streamed_log += f"\nERROR: {exc}\n"
        yield _trim_stream_log(streamed_log), {"ok": False, "error": str(exc), "message": "关联分析失败"}
        return
    result = holder.get("result", {})
    progress(1.0, desc="关联分析完成")
    streamed_log += f"\n[{result.get('message', '关联分析结束')}]\n"
    yield _trim_stream_log(streamed_log), {"ok": bool(result.get("ok")), "message": str(result.get("message", "关联分析结束")), "rules_csv": str(result.get("rules_csv", "") or ""), "rule_count": int(result.get("rule_count", 0) or 0)}

def finalize_fpgrowth_ui(run_meta: dict[str, Any] | None):
    meta = run_meta or {}
    if not meta.get("ok"):
        error = meta.get("error") or meta.get("message") or "未生成关联规则"
        return f"❌ {error}", pd.DataFrame(), [], _summary_markdown(), get_output_files(), pd.DataFrame(), 1, f"运行失败：{error}", "关联规则"
    preview, page, table_status = _page_result("关联规则", 1, PREVIEW_ROWS, "")
    rule_count = int(meta.get("rule_count", len(preview)) or len(preview))
    rules_csv = str(meta.get("rules_csv", "") or "")
    files = [rules_csv] if rules_csv and Path(rules_csv).exists() else []
    status = f"✅ {meta.get('message', '关联分析完成')} 共生成 {rule_count} 条规则；下方预览前 {len(preview)} 行。"
    return status, preview, files, _summary_markdown(get_latest_summary()), get_output_files(), preview, page, table_status, "关联规则"

def run_reslotting_ui(whn, plant, start_date, end_date, min_support, ship_upload, lx03_upload, bin_upload, aisle_priority, reserved_bins, rules, progress=gr.Progress()):
    cfg = _build_config(whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules)
    runtime_lx03, runtime_bins = _selected_upload(lx03_upload), _selected_upload(bin_upload)
    if not runtime_lx03:
        before_plot, after_plot = get_layout_plots()
        message = "请先在 01 配置与数据源上传 LX03 库位数据。"
        return f"❌ 智能调仓失败：{message}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), before_plot, after_plot, f"[智能调仓]\nERROR: {message}", get_output_files(), _summary_markdown(), get_output_files(), pd.DataFrame(), 1, f"运行失败：{message}", "移动清单"
    try:
        result = run_reslotting_pipeline(cfg, progress=lambda v, d: progress(v, desc=d), lx03_path=runtime_lx03, bin_settings_path=runtime_bins)
        previews = result.get("previews", {})
        summary = get_latest_summary()
        plots = result.get("plots", {})
        before_plot, after_plot = plots.get("before"), plots.get("after")
        if before_plot is None or after_plot is None: before_plot, after_plot = get_layout_plots()
        full_table, page, table_status = _page_result("移动清单", 1, PREVIEW_ROWS, "")
        counts = result.get("summary", {})
        status = f"✅ 调仓方案生成完成：关联组 {counts.get('identified_groups', 0)}，调仓建议 {counts.get('adjustment_rows', 0)}，实际移动 {counts.get('move_rows', 0)}，跳过 {counts.get('skip_rows', 0)}。各结果表最多预览前 {PREVIEW_ROWS} 行。"
        return status, previews.get("adjustments", pd.DataFrame()), previews.get("move_list", pd.DataFrame()), previews.get("skips", pd.DataFrame()), previews.get("group_tasks", pd.DataFrame()), previews.get("group_material_map", pd.DataFrame()), before_plot, after_plot, _combined_log("智能调仓", result), result.get("files", get_output_files()), _summary_markdown(summary), get_output_files(), full_table, page, table_status, "移动清单"
    except Exception as exc:
        before_plot, after_plot = get_layout_plots()
        return f"❌ 智能调仓失败：{exc}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), before_plot, after_plot, f"[智能调仓]\nERROR: {exc}", get_output_files(), _summary_markdown(), get_output_files(), pd.DataFrame(), 1, f"运行失败：{exc}", "移动清单"

def load_existing_slotting_ui():
    before_plot, after_plot = get_layout_plots()
    adjustments = _page_result("调仓建议", 1, PREVIEW_ROWS, "")[0]
    moves = _page_result("移动清单", 1, PREVIEW_ROWS, "")[0]
    skips = _page_result("跳过记录", 1, PREVIEW_ROWS, "")[0]
    tasks = _page_result("关联组任务", 1, PREVIEW_ROWS, "")[0]
    mapping = _page_result("物料-关联组映射", 1, PREVIEW_ROWS, "")[0]
    if any(not x.empty for x in (adjustments, moves, skips, tasks, mapping)): status = f"已载入 output/ 中的已有结果；每张表最多显示前 {PREVIEW_ROWS} 行。"
    else: status = "output/ 中暂无调仓结果。"
    files = get_output_files()
    return status, adjustments, moves, skips, tasks, mapping, before_plot, after_plot, files, _summary_markdown(), files

def refresh_results_ui():
    table, page, status = _page_result("移动清单", 1, PREVIEW_ROWS, "")
    return _summary_markdown(), get_output_files(), table, page, status, "移动清单"

def build_app() -> gr.Blocks:
    cfg = load_config()
    br = cfg.get("business_rules", {})
    initial_source_status = "### 数据源状态\n- ❌ 历史发货数据：未上传\n- ❌ LX03 库位数据：未上传\n- ✅ 库位设定：未上传时根据 LX03 自动生成"
    before_plot, after_plot = get_layout_plots()
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown("# 关联调仓解决方案\n按 **配置与数据源 → 关联分析 → 智能调仓 → 结果导出** 的顺序执行。")
        summary_md = gr.Markdown(_summary_markdown())
        with gr.Tabs():
            with gr.Tab("01 配置与数据源"):
                gr.Markdown("## 数据源\n拖入或点击上传本次运行所需的数据文件。历史发货数据和 LX03 为必选；库位设定可选，未上传时会根据 LX03 自动生成。上传文件只对当前页面会话生效，不写入 `config.json`。")
                with gr.Row():
                    with gr.Column():
                        ship_upload = gr.File(label="拖入/上传历史发货数据", file_types=[".csv", ".xlsx", ".xls"], type="filepath")
                        gr.Markdown("**历史发货数据（`ship.csv`）**：用于 FP-Growth 的 TO / 物料历史明细。至少需要 `TO number`、`Material` 两列。")
                    with gr.Column():
                        lx03_upload = gr.File(label="拖入/上传 LX03", file_types=[".xlsx", ".xls", ".csv"], type="filepath")
                        gr.Markdown("**SAP LX03 库位数据**：库存与仓位明细。程序会筛选 `仓储类型 = PD3`，并读取 `物料`、`仓位`、`总库存量` 等字段。")
                    with gr.Column():
                        bin_upload = gr.File(label="拖入/上传库位设定（可选）", file_types=[".xlsx", ".xls", ".csv"], type="filepath")
                        gr.Markdown("**库位设定（可选）**：用于覆盖物料—库位基础映射；不上传时直接根据当前 LX03 的 PD3 数据自动生成。")
                with gr.Row():
                    check_sources = gr.Button("检查数据源")
                    preview_sources = gr.Button("读取并预览数据")
                source_status = gr.Markdown(initial_source_status)
                gr.Markdown("### 数据预览（最多前 200 行）")
                with gr.Tabs():
                    with gr.Tab("历史发货数据"): ship_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="发货数据预览", max_height=420)
                    with gr.Tab("LX03 / PD3"): lx03_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="PD3 库位预览", max_height=420)
                    with gr.Tab("库位设定"): bin_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="库位设定预览", max_height=420)
                data_preview_status = gr.Markdown("尚未读取数据。")
                gr.Markdown("---\n## 业务配置")
                with gr.Row():
                    whn = gr.Textbox(value=cfg.get("whn", ""), label="WHN")
                    plant = gr.Textbox(value=cfg.get("plant", ""), label="Plant")
                    start_date = gr.Textbox(value=cfg.get("start_date", ""), label="开始日期")
                    end_date = gr.Textbox(value=cfg.get("end_date", ""), label="结束日期")
                    min_support = gr.Number(value=float(cfg.get("min_support", 0.0003)), label="最小支持度", minimum=0.000001, maximum=1, step=0.0001)
                with gr.Row():
                    aisle_priority = gr.Textbox(value=",".join(br.get("aisle_priority", [])), label="通道前端优先顺序", info="逗号分隔，例如 07,06")
                    reserved_bins = gr.Textbox(value="\n".join(br.get("reserved_bin_patterns", [])), label="保留库位模式", lines=3, info="每行或逗号分隔")
                gr.Markdown("### 物料通道规则")
                rules_df = gr.Dataframe(value=_rules_frame(cfg), headers=RULE_COLUMNS, datatype="str", interactive=True, label="业务规则", show_row_numbers=True, wrap=True, max_height=360)
                with gr.Row():
                    add_rule = gr.Button("添加空规则")
                    reload_config = gr.Button("重新载入配置")
                    save_config_btn = gr.Button("保存配置", variant="primary")
                config_status = gr.Markdown("")
            with gr.Tab("02 数据与关联分析"):
                gr.Markdown("## 关联分析\n执行 FP-Growth 后，完整结果写入 `output/`；运行期间显示标准任务进度，**只有关联分析日志会随算法输出实时刷新**。完成后再一次性显示前 200 行规则和输出文件。")
                run_fp = gr.Button("执行关联分析", variant="primary", size="lg")
                fp_status = gr.Markdown("尚未执行关联分析。")
                fp_rules_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="关联规则预览（最多前 200 行）", max_height=520, wrap=False)
                with gr.Row():
                    fp_files = gr.File(value=[], file_count="multiple", label="关联分析输出文件", interactive=False)
                    fp_log = gr.Textbox(value="", label="关联分析日志", lines=12, max_lines=24, interactive=False, autoscroll=True)
                fp_run_state = gr.State(value=None)
            with gr.Tab("03 智能调仓"):
                gr.Markdown("## 智能调仓\n请先完成关联分析。调仓结束后会立即显示主要结果表的前 200 行，并生成可缩放、平移、悬浮查看的 Plotly 交互布局；SVG 仍作为矢量文件导出。")
                with gr.Row():
                    run_slotting = gr.Button("生成调仓方案", variant="primary", size="lg")
                    load_existing = gr.Button("载入已有调仓结果")
                slot_status = gr.Markdown("尚未执行智能调仓。")
                gr.Markdown("### 调仓结果预览（每表最多前 200 行）")
                with gr.Tabs():
                    with gr.Tab("调仓建议"): adjustments_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="调仓建议", max_height=500)
                    with gr.Tab("移动清单"): moves_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="移动清单", max_height=500)
                    with gr.Tab("跳过记录"): skips_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="跳过记录", max_height=500)
                    with gr.Tab("关联组任务"): tasks_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="关联组任务", max_height=500)
                    with gr.Tab("物料-关联组映射"): mapping_preview = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="物料-关联组映射", max_height=500)
                gr.Markdown("### Plotly 交互式布局预览")
                gr.Markdown("可使用工具栏缩放/框选，拖动画布平移，并将鼠标悬停在物料上查看库位、关联组和调仓信息。")
                with gr.Row():
                    with gr.Column(): before_layout_plot = gr.Plot(value=before_plot, label="调仓前")
                    with gr.Column(): after_layout_plot = gr.Plot(value=after_plot, label="调仓后")
                with gr.Row():
                    slot_files = gr.File(value=get_output_files(), file_count="multiple", label="调仓输出文件", interactive=False)
                    slot_log = gr.Textbox(value="", label="智能调仓日志", lines=12, max_lines=24, interactive=False, autoscroll=True)
            with gr.Tab("04 完整结果与导出"):
                gr.Markdown("## 完整结果\n这里采用服务端分页。浏览器每次只接收当前页数据，完整结果仍保存在 `output/` 文件中。")
                with gr.Row():
                    table_name = gr.Dropdown(choices=list(TABLE_MAP.keys()), value="移动清单", label="结果表")
                    search = gr.Textbox(label="搜索", placeholder="物料 / 通道 / 原因 / 关联组…")
                    page_size = gr.Dropdown(choices=PAGE_SIZE_CHOICES, value=PREVIEW_ROWS, label="每页行数")
                result_table = gr.Dataframe(value=pd.DataFrame(), interactive=False, label="结果分页预览", max_height=560, wrap=False)
                with gr.Row(equal_height=True):
                    prev_btn = gr.Button("上一页", scale=2)
                    page_no = gr.Number(value=1, precision=0, minimum=1, placeholder="页码", show_label=False, container=False, min_width=90, scale=1)
                    next_btn = gr.Button("下一页", scale=2)
                    refresh_results = gr.Button("刷新结果", scale=2)
                page_status = gr.Markdown("尚未载入结果。")
                output_files = gr.File(value=get_output_files(), file_count="multiple", label="全部输出文件", interactive=False)
        check_sources.click(check_sources_ui, inputs=[ship_upload, lx03_upload, bin_upload], outputs=source_status)
        preview_sources.click(preview_sources_ui, inputs=[ship_upload, lx03_upload, bin_upload], outputs=[data_preview_status, ship_preview, lx03_preview, bin_preview])
        for source_component in (ship_upload, lx03_upload, bin_upload):
            source_component.change(check_sources_ui, inputs=[ship_upload, lx03_upload, bin_upload], outputs=source_status, show_progress="hidden")
        add_rule.click(add_rule_row, inputs=rules_df, outputs=rules_df)
        reload_config.click(load_config_ui, outputs=[whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules_df, config_status])
        save_config_btn.click(save_config_ui, inputs=[whn, plant, start_date, end_date, min_support, aisle_priority, reserved_bins, rules_df], outputs=[rules_df, config_status])
        common_inputs = [whn, plant, start_date, end_date, min_support, ship_upload, lx03_upload, bin_upload, aisle_priority, reserved_bins, rules_df]
        fp_stream_event = run_fp.click(stream_fpgrowth_log, inputs=common_inputs, outputs=[fp_log, fp_run_state], show_progress="full", show_progress_on=[fp_files, summary_md])
        fp_stream_event.then(finalize_fpgrowth_ui, inputs=fp_run_state, outputs=[fp_status, fp_rules_preview, fp_files, summary_md, output_files, result_table, page_no, page_status, table_name], show_progress="hidden")
        run_slotting.click(run_reslotting_ui, inputs=common_inputs, outputs=[slot_status, adjustments_preview, moves_preview, skips_preview, tasks_preview, mapping_preview, before_layout_plot, after_layout_plot, slot_log, slot_files, summary_md, output_files, result_table, page_no, page_status, table_name])
        load_existing.click(load_existing_slotting_ui, outputs=[slot_status, adjustments_preview, moves_preview, skips_preview, tasks_preview, mapping_preview, before_layout_plot, after_layout_plot, slot_files, summary_md, output_files])
        for trigger in (table_name.change, page_size.change, search.submit, page_no.submit):
            trigger(_page_result, inputs=[table_name, page_no, page_size, search], outputs=[result_table, page_no, page_status])
        prev_btn.click(prev_page, inputs=[table_name, page_no, page_size, search], outputs=[result_table, page_no, page_status])
        next_btn.click(next_page, inputs=[table_name, page_no, page_size, search], outputs=[result_table, page_no, page_status])
        refresh_results.click(refresh_results_ui, outputs=[summary_md, output_files, result_table, page_no, page_status, table_name])
        demo.load(refresh_results_ui, outputs=[summary_md, output_files, result_table, page_no, page_status, table_name])
    return demo

def main() -> None:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_app().queue(default_concurrency_limit=1)
    demo.launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser, show_error=True, share=False, allowed_paths=[str(OUTPUT_DIR)])

if __name__ == "__main__":
    main()