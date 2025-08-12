# app.py — Milwaukee Small‑Bay Industrial Finder (MVP)
# ---------------------------------------------------
# What it does
# - Loads Milwaukee County parcels + optional address points + optional building footprints
# - Filters for likely small‑bay industrial via zoning/land-use + size heuristics
# - Adds a simple candidate score and a multi‑tenant proxy (# of address points per parcel)
# - Lets you visually QC on a map (with legal USGS imagery) and export a call sheet CSV
#
# Data you’ll need (download once; paths set in the sidebar):
# 1) Parcels w/ attributes (GeoPackage/GeoJSON recommended)
#    Source: Milwaukee County LIO “Cadastral Dataset / Parcels with Property Info”
#    Hub: https://data-mclio.hub.arcgis.com/ (download > convert to .gpkg if needed)
# 2) (Optional) Address Points layer (to estimate suite/address counts per parcel)
#    Hub item: “Milwaukee County Address Points”
# 3) (Optional) Building footprints (Microsoft US Building Footprints for WI) to compute building_sf
#    https://planetarycomputer.microsoft.com/dataset/ms-buildings (clip to Milwaukee)
#
# Legal imagery note: the basemap uses **USGS ImageryOnly** (public domain). Do not use Google/Bing tiles
# for data derivation per their TOS.
#
# Run
#   $ pip install streamlit geopandas shapely pydeck pandas numpy requests fiona pyproj rtree
#   $ streamlit run app.py

import io
import json
import math
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from shapely.affinity import rotate
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="Milwaukee Small‑Bay Finder", layout="wide")
st.title("Milwaukee Small‑Bay Industrial Finder — MVP")

# ----------------------------
# Sidebar — data inputs
# ----------------------------
st.sidebar.header("Data inputs")
parcels_file = st.sidebar.file_uploader(
    "Parcels (GeoPackage/GeoJSON)",
    type=["gpkg", "geojson", "json"],
    help="Download from Milwaukee County LIO > Cadastral Dataset. Prefer a clipped export for City of Milwaukee to keep it light.",
)

addrpts_file = st.sidebar.file_uploader(
    "(Optional) Address Points (GeoPackage/GeoJSON)",
    type=["gpkg", "geojson", "json"],
    help="Used to count addresses per parcel (multi‑tenant proxy).",
)

bldg_file = st.sidebar.file_uploader(
    "(Optional) Building Footprints (GeoPackage/GeoJSON)",
    type=["gpkg", "geojson", "json"],
    help="If missing building_sf on parcels, we’ll compute from footprints joined to parcels.",
)

# ----------------------------
# Sidebar — heuristics
# ----------------------------
st.sidebar.header("Filters & scoring")
min_sf = st.sidebar.number_input("Min building sf", value=10000, step=1000)
max_sf = st.sidebar.number_input("Max building sf", value=200000, step=1000)

st.sidebar.markdown("**Industrial zoning to include** (Milwaukee code 295‑801):")
use_io = st.sidebar.checkbox("IO (Industrial‑Office)", value=True)
use_il = st.sidebar.checkbox("IL (Industrial‑Light)", value=True)
use_ic = st.sidebar.checkbox("IC (Industrial‑Commercial)", value=True)
use_im = st.sidebar.checkbox("IM (Industrial‑Mixed)", value=False)
use_ih = st.sidebar.checkbox("IH (Industrial‑Heavy)", value=False)

min_addrpts = st.sidebar.number_input("Min address points per parcel (proxy for multi‑tenant)", value=2, min_value=0, step=1)

# scoring weights
st.sidebar.subheader("Scoring weights")
w_zoning = st.sidebar.slider("Zoning match", 0, 5, 3)
w_size_mid = st.sidebar.slider("Ideal size band (20k‑120k)", 0, 5, 2)
w_addrpts = st.sidebar.slider("Address points \u2265 3", 0, 5, 2)
w_depth = st.sidebar.slider("Shallow rectangle depth ratio", 0, 3, 1)
score_threshold = st.sidebar.slider("Candidate score threshold", 0, 10, 5)

# ----------------------------
# Helper functions
# ----------------------------

def read_any_vector(file) -> gpd.GeoDataFrame:
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".gpkg"):
        return gpd.read_file(file)
    elif name.endswith(".geojson") or name.endswith(".json"):
        return gpd.read_file(file)
    else:
        raise ValueError("Unsupported format. Use GeoPackage (.gpkg) or GeoJSON.")


def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        # assume WGS84
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(epsg)


def guess_columns(gdf: gpd.GeoDataFrame):
    cols = {c.lower(): c for c in gdf.columns}
    guess = {
        "apn": cols.get("taxkey") or cols.get("apn") or cols.get("parcelid") or cols.get("parcel_id"),
        "owner": cols.get("owner") or cols.get("ownernme1") or cols.get("ownernam") or cols.get("owner_name"),
        "mail_addr": cols.get("mailaddr") or cols.get("owneraddr") or cols.get("mailing_address") or cols.get("mailadd1"),
        "mail_city": cols.get("mailcity") or cols.get("ownercity") or cols.get("mail_city"),
        "mail_state": cols.get("mailst") or cols.get("ownerst") or cols.get("mail_state"),
        "mail_zip": cols.get("mailzip") or cols.get("ownerzip") or cols.get("mail_zip"),
        "situs": cols.get("situsaddr") or cols.get("siteaddr") or cols.get("property_address") or cols.get("situs_addr"),
        "zoning": cols.get("zoning") or cols.get("zone_cd") or cols.get("zoning_code") or cols.get("zoning_d"),
        "land_use": cols.get("landuse") or cols.get("land_use") or cols.get("mprop_landuse") or cols.get("landuse_g"),
        "bldg_sf": cols.get("bldg_sqft") or cols.get("bldg_area") or cols.get("imprv_area") or cols.get("impr_sqft") or cols.get("impsf")
    }
    return guess


def min_rotated_rect_depth_ratio(poly: Polygon) -> Optional[float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        # rectangle has 5 points with first==last
        edges = [math.dist(coords[i], coords[i+1]) for i in range(4)]
        edges.sort()
        short, long = edges[0], edges[2]
        # ratio >1 indicates shallow depth relative to frontage-ish
        return long / max(short, 1e-6)
    except Exception:
        return None


def add_building_sf_from_footprints(parcels: gpd.GeoDataFrame, bldgs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # spatial join (intersects), then sum footprint area per parcel
    bldgs = ensure_crs(bldgs, parcels.crs.to_epsg() if parcels.crs else 3857)
    bldgs["foot_area"] = bldgs.geometry.area
    joined = gpd.sjoin(bldgs[["foot_area", "geometry"]], parcels[["geometry"]], how="inner", predicate="intersects")
    sums = joined.groupby(joined.index_right)["foot_area"].sum()
    parcels = parcels.copy()
    parcels["bldg_sf_calc"] = sums
    return parcels


def zone_filter(z: Optional[str]) -> bool:
    if not z or not isinstance(z, str):
        return False
    z = z.upper()
    allow = []
    if use_io: allow += ["IO", "IO1", "IO2"]
    if use_il: allow += ["IL", "IL1", "IL2"]
    if use_ic: allow += ["IC"]
    if use_im: allow += ["IM"]
    if use_ih: allow += ["IH"]
    return any(z.startswith(code) for code in allow)


def size_band_points(sf: float) -> int:
    if sf is None or np.isnan(sf):
        return 0
    if 20000 <= sf <= 120000:
        return w_size_mid
    return 1 if (min_sf <= sf <= max_sf) else 0


def compute_score(row) -> int:
    score = 0
    if row.get("zoning_ok", False):
        score += w_zoning
    score += size_band_points(row.get("bldg_sf"))
    if row.get("addr_count", 0) >= 3:
        score += w_addrpts
    # shallow depth proxy from rotated rect: long/short > 1.2
    if row.get("depth_ratio") and row["depth_ratio"] > 1.2:
        score += w_depth
    return int(score)

# ----------------------------
# Load data
# ----------------------------
parcels = read_any_vector(parcels_file)
if parcels is None:
    st.info("⬅️ Load a parcels file to begin.")
    st.stop()

parcels = ensure_crs(parcels)
colmap = guess_columns(parcels)

# (Optional) Address points -> address count per parcel
addr_counts = None
if addrpts_file is not None:
    addr = read_any_vector(addrpts_file)
    if addr is not None and not addr.empty:
        addr = ensure_crs(addr, parcels.crs.to_epsg())
        # fast spatial join, then count
        sjoin = gpd.sjoin(addr[["geometry"]], parcels[["geometry"]], how="inner", predicate="within")
        addr_counts = sjoin.groupby(sjoin.index_right).size()

# (Optional) Building footprints -> computed building_sf
if bldg_file is not None:
    bldgs = read_any_vector(bldg_file)
    if bldgs is not None and not bldgs.empty:
        parcels = add_building_sf_from_footprints(parcels, bldgs)

# ----------------------------
# Feature engineering
# ----------------------------
parcels = parcels.copy()

# Choose building_sf
bldg_sf_col = colmap.get("bldg_sf")
if bldg_sf_col in parcels.columns:
    parcels["bldg_sf"] = pd.to_numeric(parcels[bldg_sf_col], errors="coerce")
else:
    parcels["bldg_sf"] = parcels.get("bldg_sf_calc")

# Zoning flag
zone_col = colmap.get("zoning") or colmap.get("land_use")
parcels["zoning_txt"] = parcels[zone_col].astype(str) if zone_col in parcels.columns else ""
parcels["zoning_ok"] = parcels["zoning_txt"].apply(zone_filter)

# Address count
if addr_counts is not None:
    parcels["addr_count"] = parcels.index.map(addr_counts).fillna(0).astype(int)
else:
    parcels["addr_count"] = 0

# Depth ratio from min rotated rectangle
parcels["depth_ratio"] = parcels.geometry.apply(min_rotated_rect_depth_ratio)

# Score
parcels["score"] = parcels.apply(compute_score, axis=1)

# Filters
mask = (
    parcels["zoning_ok"]
    & parcels["bldg_sf"].between(min_sf, max_sf, inclusive="both")
    & (parcels["addr_count"] >= min_addrpts)
)

candidates = parcels[mask].copy()
candidates = candidates.sort_values("score", ascending=False)

st.markdown(f"**Candidates: {len(candidates):,} parcels** meet your current filters.")

# ----------------------------
# Map
# ----------------------------
# Center on Milwaukee
initial_view = pdk.ViewState(latitude=43.0389, longitude=-87.9065, zoom=11, pitch=0)

# USGS ImageryOnly tiles (public domain)
usgs_tiles = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

tile_layer = pdk.Layer(
    "TileLayer",
    data=usgs_tiles,
    min_zoom=0,
    max_zoom=19,
    tile_size=256,
)

# Candidate polygons
# Convert polygons to GeoJSON-like features for pydeck
cand_geojson = json.loads(candidates.to_crs(4326).to_json())

poly_layer = pdk.Layer(
    "GeoJsonLayer",
    data=cand_geojson,
    stroked=True,
    filled=True,
    get_fill_color="[score >= %d ? 255 : 200, 100, 0, 60]" % score_threshold,
    get_line_color=[255, 140, 0],
    get_line_width=1,
    pickable=True,
)

r = pdk.Deck(layers=[tile_layer, poly_layer], initial_view_state=initial_view, map_provider=None)
st.pydeck_chart(r)

st.caption("Imagery: USGS The National Map (USGSImageryOnly). Parcels: Milwaukee County LIO.")

# ----------------------------
# Call sheet export
# ----------------------------
owner_col = colmap.get("owner")
mail_cols = [colmap.get("mail_addr"), colmap.get("mail_city"), colmap.get("mail_state"), colmap.get("mail_zip")]

export_cols = {
    "APN": colmap.get("apn"),
    "Property Address": colmap.get("situs"),
    "Zoning": "zoning_txt",
    "Building SF": "bldg_sf",
    "Address Count": "addr_count",
    "Depth Ratio": "depth_ratio",
    "Score": "score",
}

if owner_col in parcels.columns:
    export_cols["Owner Name"] = owner_col
if any(c in parcels.columns for c in mail_cols if c):
    export_cols["Owner Mailing Address"] = colmap.get("mail_addr")
    export_cols["Owner Mailing City"] = colmap.get("mail_city")
    export_cols["Owner Mailing State"] = colmap.get("mail_state")
    export_cols["Owner Mailing Zip"] = colmap.get("mail_zip")

export_df = candidates[[v for v in export_cols.values() if v in candidates.columns]].copy()
export_df.columns = [k for k, v in export_cols.items() if v in candidates.columns]

st.subheader("Export call sheet")
st.dataframe(export_df.head(50))

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="milwaukee_small_bay_calls.csv", mime="text/csv")

st.markdown("---")
with st.expander("Notes / Tips"):
    st.markdown(
        """
        **Zoning filters** use Milwaukee’s industrial districts (IO, IL, IC, IM, IH) defined in the City zoning code.
        If your parcel file uses land‑use instead of zoning, turn on IC/IM to avoid filtering out light industrial corridors.

        **Multi‑tenant proxy** uses address points; if you don’t provide them, set the minimum to 0.

        **Building size** comes from a `bldg_sf` column if present; otherwise from joined building footprints.

        **Imagery** is USGS ImageryOnly (public domain). You can substitute NAIP tiles or Esri imagery you’re licensed to use.

        **Next**: add Secretary of State lookups for LLC principals, enrich phones, and add a review workflow (Yes/No/Unsure) that writes back to a SQLite table.
        """
    )
