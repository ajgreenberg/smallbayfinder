# app.py — Milwaukee Small‑Bay Industrial Finder (MVP)
# ---------------------------------------------------
# Zero‑download mode for Milwaukee: the app can auto‑fetch parcels and
# address points directly from the City of Milwaukee ArcGIS services.
# You can still upload your own files if you prefer.
# Imagery uses USGS (public domain). No Google/Bing scraping.
#
# Quick start
#   pip install streamlit geopandas shapely pyproj pandas numpy pydeck requests rtree
#   # (optional, smoother file I/O on cloud)
#   pip install pyogrio
#   streamlit run app.py

import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit as st
import pydeck as pdk
import requests

# Prefer pyogrio if present
try:
    gpd.options.io_engine = "pyogrio"
except Exception:
    pass

st.set_page_config(page_title="Milwaukee Small‑Bay Finder", layout="wide")
st.title("Milwaukee Small‑Bay Industrial Finder — MVP")

# ---------------------------------
# Constants — City of Milwaukee GIS
# ---------------------------------
PARCELS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2"
ADDRPTS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22"
USGS_TILES = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

# ---------------------------------
# Widgets (use explicit keys to avoid duplicate IDs)
# ---------------------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio(
    "Data source mode",
    ["Auto‑fetch (no downloads)", "Upload files"],
    index=0,
    key="mode_radio",
)

st.sidebar.header("Filters & scoring")
min_sf = st.sidebar.number_input("Min building sf", value=10000, step=1000, key="min_sf")
max_sf = st.sidebar.number_input("Max building sf", value=200000, step=1000, key="max_sf")

st.sidebar.markdown("**Industrial zoning to include (Milwaukee 295‑801):**")
use_io = st.sidebar.checkbox("IO (Industrial‑Office)", value=True, key="zone_io")
use_il = st.sidebar.checkbox("IL (Industrial‑Light)", value=True, key="zone_il")
use_ic = st.sidebar.checkbox("IC (Industrial‑Commercial)", value=True, key="zone_ic")
use_im = st.sidebar.checkbox("IM (Industrial‑Mixed)", value=False, key="zone_im")
use_ih = st.sidebar.checkbox("IH (Industrial‑Heavy)", value=False, key="zone_ih")

min_addrpts = st.sidebar.number_input(
    "Min address points per parcel (multi‑tenant proxy)", value=0, min_value=0, step=1, key="min_addrpts"
)

st.sidebar.subheader("Scoring weights")
w_zoning = st.sidebar.slider("Zoning match", 0, 5, 3, key="w_zoning")
w_size_mid = st.sidebar.slider("Ideal size band (20k‑120k)", 0, 5, 2, key="w_size_mid")
w_addrpts = st.sidebar.slider("Address points ≥ 3", 0, 5, 2, key="w_addrpts")
w_depth = st.sidebar.slider("Shallow rectangle depth", 0, 3, 1, key="w_depth")
score_threshold = st.sidebar.slider("Candidate score threshold", 0, 10, 5, key="score_threshold")

# ---------------------------------
# Helpers
# ---------------------------------

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs=4326)
    return gdf.to_crs(epsg)


def min_rotated_rect_depth_ratio(poly: Polygon) -> Optional[float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) ** 0.5 for i in range(4)]
        edges.sort()
        short, long = edges[0], edges[2]
        return long / max(short, 1e-6)
    except Exception:
        return None


def allow_zoning_prefixes():
    allow = []
    if use_io: allow += ["IO"]
    if use_il: allow += ["IL"]
    if use_ic: allow += ["IC"]
    if use_im: allow += ["IM"]
    if use_ih: allow += ["IH"]
    return allow


def build_where_clause():
    # ZONING may contain letter+digit (e.g., IO1, IL2) — use LIKE on prefixes
    prefixes = allow_zoning_prefixes()
    if prefixes:
        likes = [f"ZONING LIKE '{p}%'" for p in prefixes]
        z_clause = "(" + " OR ".join(likes) + ")"
    else:
        z_clause = "1=1"
    sf_clause = f"BLDG_AREA >= {min_sf} AND BLDG_AREA <= {max_sf}"
    return f"{z_clause} AND {sf_clause}"

@st.cache_data(show_spinner=False)
def fetch_arcgis_geojson(layer_url: str, where: str, out_fields: str = "*", geometry: Optional[dict] = None) -> gpd.GeoDataFrame:
    """Fetch GeoJSON from an ArcGIS layer that supports geoJSON and return a GeoDataFrame (EPSG:4326)."""
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }
    if geometry:
        params.update({
            "geometry": json.dumps(geometry),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })
    url = layer_url.rstrip('/') + "/query"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    gj = r.json()
    if not isinstance(gj, dict) or "features" not in gj:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(gj.get("features", []), crs="EPSG:4326")
    return gdf

# ---------------------------------
# Data loading — two modes
# ---------------------------------
parcels = None
addrpts = None

if mode == "Auto‑fetch (no downloads)":
    st.sidebar.subheader("Auto‑fetch options")
    extent_choice = st.sidebar.selectbox(
        "Area",
        ["Citywide (Industrial filter)", "Menomonee Valley demo", "Airport/College Ave demo"],
        index=0,
        key="extent_choice",
    )

    # Simple demo extents (lon/lat envelopes)
    extent = None
    if extent_choice == "Menomonee Valley demo":
        extent = {"xmin": -87.961, "ymin": 43.015, "xmax": -87.920, "ymax": 43.035}
    elif extent_choice == "Airport/College Ave demo":
        extent = {"xmin": -87.944, "ymin": 42.915, "xmax": -87.903, "ymax": 42.940}

    if st.sidebar.button("Fetch parcels", key="fetch_parcels_btn"):
        with st.spinner("Fetching parcels from City of Milwaukee…"):
            where = build_where_clause()
            out_fields = (
                "TAXKEY,OWNER_NAME_1,OWNER_MAIL_ADDR,OWNER_CITY_STATE,OWNER_ZIP,"
                "ZONING,LAND_USE,BLDG_AREA,UNIT,HOUSE_NR_LO,SDIR,STREET,STTYPE"
            )
            parcels = fetch_arcgis_geojson(PARCELS_LAYER, where, out_fields, geometry=extent)
        st.success(f"Loaded {len(parcels):,} parcels.")

    fetch_pts = st.sidebar.checkbox(
        "Also fetch Address Points for multi‑tenant signal", value=True, key="fetch_addrpts_chk"
    )
    if fetch_pts and parcels is not None and not parcels.empty:
        with st.spinner("Fetching Address Points…"):
            addrpts = fetch_arcgis_geojson(ADDRPTS_LAYER, "1=1", out_fields="*", geometry=extent)
        st.success(f"Loaded {len(addrpts):,} address points.")

else:
    st.sidebar.subheader("Upload files")
    parcels_file = st.sidebar.file_uploader(
        "Parcels (GeoPackage/GeoJSON/Shapefile)",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_parcels",
    )
    addrpts_file = st.sidebar.file_uploader(
        "(Optional) Address Points",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_addrpts",
    )
    bldg_file = st.sidebar.file_uploader(
        "(Optional) Building Footprints",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_bldgs",
    )

    def read_any_vector(file) -> gpd.GeoDataFrame:
        if file is None:
            return None
        return gpd.read_file(file)

    if parcels_file:
        parcels = read_any_vector(parcels_file)
    if addrpts_file:
        addrpts = read_any_vector(addrpts_file)
    if bldg_file:
        bldgs = read_any_vector(bldg_file)
        if bldgs is not None and not bldgs.empty and parcels is not None and not parcels.empty:
            bldgs = bldgs.to_crs(parcels.crs or "EPSG:4326")
            bldgs["foot_area"] = bldgs.geometry.area
            joined = gpd.sjoin(bldgs[["foot_area", "geometry"]], parcels[["geometry"]], how="inner", predicate="intersects")
            sums = joined.groupby(joined.index_right)["foot_area"].sum()
            parcels = parcels.copy()
            parcels["BLDG_AREA"] = sums

# Guard — need parcels to proceed
if parcels is None or len(parcels) == 0:
    st.info("⬅️ Use **Auto‑fetch** and click *Fetch parcels* (recommended), or upload a parcels file.")
    st.stop()

# ---------------------------------
# Feature engineering
# ---------------------------------
cols_lower = {c.lower(): c for c in parcels.columns}
getc = lambda *names: next((cols_lower[n] for n in names if n in cols_lower), None)

apn_col = getc("taxkey", "apn", "parcelid", "parcel_id")
owner1_col = getc("owner_name_1", "owner", "ownernme1")
mail_addr_col = getc("owner_mail_addr", "mailaddr", "owneraddr", "mailing_address")
mail_cityst_col = getc("owner_city_state", "mail_city_state")
mail_zip_col = getc("owner_zip", "mailzip")
zoning_col = getc("zoning")
landuse_col = getc("land_use", "landuse")
bldg_col = getc("bldg_area", "imprv_sqft", "bldg_sqft", "imprv_area")

# Fill missing BLDG_AREA with NaN so filters work
if bldg_col is None:
    parcels["BLDG_AREA"] = np.nan
    bldg_col = "BLDG_AREA"

# Address counts via spatial join
addr_count = pd.Series(0, index=parcels.index)
if addrpts is not None and not addrpts.empty:
    try:
        addrpts = addrpts.to_crs(parcels.crs or "EPSG:4326")
        parcels = parcels.to_crs(addrpts.crs)
        sjoin = gpd.sjoin(addrpts[["geometry"]], parcels[["geometry"]], how="inner", predicate="within")
        counts = sjoin.groupby(sjoin.index_right).size()
        addr_count = addr_count.add(counts, fill_value=0).astype(int)
    except Exception:
        pass

work = parcels.copy()
work["addr_count"] = addr_count

# Zoning flag (prefix check)
allow_prefixes = allow_zoning_prefixes()
if zoning_col is not None and allow_prefixes:
    ztxt = work[zoning_col].astype(str).str.upper().fillna("")
    work["zoning_ok"] = ztxt.str.startswith(tuple(allow_prefixes)) | ztxt.isin(allow_prefixes)
else:
    work["zoning_ok"] = True  # if no zoning, don’t exclude

# Size filter & depth ratio
work["bldg_sf"] = pd.to_numeric(work[bldg_col], errors="coerce")
work_3857 = ensure_crs(work)
work["depth_ratio"] = work_3857.geometry.apply(min_rotated_rect_depth_ratio)

# Scoring

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
    if row.get("depth_ratio") and row["depth_ratio"] > 1.2:
        score += w_depth
    return int(score)

work["score"] = work.apply(compute_score, axis=1)

# Effective min address points (auto‑relax if none loaded)
addrpts_loaded = addrpts is not None and len(addrpts) > 0
effective_min_addrpts = min_addrpts if addrpts_loaded else 0
if not addrpts_loaded:
    st.warning("No address points loaded — using Min address points = 0 for filtering.")

# Apply hard filters
mask = (
    work["bldg_sf"].between(min_sf, max_sf, inclusive="both") &
    work["zoning_ok"] &
    (work["addr_count"] >= effective_min_addrpts)
)

cand = work[mask].copy().sort_values("score", ascending=False)

if len(cand) == 0:
    st.error("No candidates yet. Try lowering Min building sf, enabling more zoning types (IC/IM/IH), or widen the area.")
else:
    st.markdown(f"**Candidates: {len(cand):,} parcels** meet your current filters.")

# ---------------------------------
# Map
# ---------------------------------
initial_view = pdk.ViewState(latitude=43.0389, longitude=-87.9065, zoom=11)

tile_layer = pdk.Layer("TileLayer", data=USGS_TILES, min_zoom=0, max_zoom=19, tile_size=256)

cand_geojson = json.loads(cand.to_crs(4326).to_json()) if len(cand) else {"type":"FeatureCollection","features":[]}
poly_layer = pdk.Layer(
    "GeoJsonLayer",
    data=cand_geojson,
    stroked=True,
    filled=True,
    get_fill_color=f"[score >= {score_threshold} ? 255 : 200, 100, 0, 60]",
    get_line_color=[255, 140, 0],
    get_line_width=1,
    pickable=True,
)

r = pdk.Deck(layers=[tile_layer, poly_layer], initial_view_state=initial_view, map_provider=None)
st.pydeck_chart(r, use_container_width=True)

st.caption("Imagery: USGS The National Map (USGSImageryOnly). Parcels & Address Points: City of Milwaukee MapServer (GeoJSON).")

# ---------------------------------
# Export call sheet
# ---------------------------------
export_cols = {
    "APN": apn_col,
    "Zoning": zoning_col,
    "Building SF": "bldg_sf",
    "Address Count": "addr_count",
    "Depth Ratio": "depth_ratio",
    "Score": "score",
    "Owner Name": owner1_col,
    "Owner Mailing Address": mail_addr_col,
    "Owner City/State": mail_cityst_col,
    "Owner Zip": mail_zip_col,
}

# Compose a property address if street fields exist
cols_lower = {c.lower(): c for c in work.columns}
parts = [cols_lower.get(p) for p in ["house_nr_lo", "sdir", "street", "sttype", "unit"] if cols_lower.get(p)]
if parts:
    addr_series = (
        work.get(cols_lower.get("house_nr_lo"), pd.Series("")).astype(str).str.replace(".0$", "", regex=True) + " " +
        work.get(cols_lower.get("sdir"), pd.Series("")).astype(str) + " " +
        work.get(cols_lower.get("street"), pd.Series("")).astype(str) + " " +
        work.get(cols_lower.get("sttype"), pd.Series("")).astype(str) +
        (" #" + work.get(cols_lower.get("unit"), pd.Series("")).astype(str)).where(work.get(cols_lower.get("unit"), pd.Series("")).astype(str).ne(""), "")
    ).str.replace(" +", " ", regex=True).str.strip()
    work["Property Address"] = addr_series
    export_cols = {"Property Address": "Property Address", **export_cols}

final_cols = [v for v in export_cols.values() if v is not None and v in work.columns]
export_df = work.loc[cand.index, final_cols].copy()
export_df.columns = [k for k, v in export_cols.items() if v is not None and v in work.columns]

st.subheader("Export call sheet")
st.dataframe(export_df.head(50))

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="milwaukee_small_bay_calls.csv", mime="text/csv", key="dl_csv")

# ---------------------------------
# Diagnostics
# ---------------------------------
with st.expander("Diagnostics"):
    st.write({
        "parcels_loaded": 0 if parcels is None else len(parcels),
        "addrpts_loaded": 0 if addrpts is None else len(addrpts),
        "candidates": len(cand),
        "filters": {
            "min_sf": min_sf,
            "max_sf": max_sf,
            "zoning_prefixes": allow_zoning_prefixes(),
            "min_addrpts_effective": effective_min_addrpts,
        },
    })
# app.py — Milwaukee Small‑Bay Industrial Finder (MVP)
# ---------------------------------------------------
# Zero‑download mode for Milwaukee: the app can auto‑fetch parcels and
# address points directly from the City of Milwaukee ArcGIS services.
# You can still upload your own files if you prefer.
# Imagery uses USGS (public domain). No Google/Bing scraping.
#
# Quick start
#   pip install streamlit geopandas shapely pyproj pandas numpy pydeck requests rtree
#   # (optional, smoother file I/O on cloud)
#   pip install pyogrio
#   streamlit run app.py

import json
import math
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit as st
import pydeck as pdk
import requests

# Prefer pyogrio if present
try:
    gpd.options.io_engine = "pyogrio"
except Exception:
    pass

st.set_page_config(page_title="Milwaukee Small‑Bay Finder", layout="wide")
st.title("Milwaukee Small‑Bay Industrial Finder — MVP")

# ---------------------------------
# Constants — City of Milwaukee GIS
# ---------------------------------
PARCELS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2"
ADDRPTS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22"
USGS_TILES = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

# ---------------------------------
# Widgets (use explicit keys to avoid duplicate IDs)
# ---------------------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio(
    "Data source mode",
    ["Auto‑fetch (no downloads)", "Upload files"],
    index=0,
    key="mode_radio",
)

st.sidebar.header("Filters & scoring")
min_sf = st.sidebar.number_input("Min building sf", value=10000, step=1000, key="min_sf")
max_sf = st.sidebar.number_input("Max building sf", value=200000, step=1000, key="max_sf")

st.sidebar.markdown("**Industrial zoning to include (Milwaukee 295‑801):**")
use_io = st.sidebar.checkbox("IO (Industrial‑Office)", value=True, key="zone_io")
use_il = st.sidebar.checkbox("IL (Industrial‑Light)", value=True, key="zone_il")
use_ic = st.sidebar.checkbox("IC (Industrial‑Commercial)", value=True, key="zone_ic")
use_im = st.sidebar.checkbox("IM (Industrial‑Mixed)", value=False, key="zone_im")
use_ih = st.sidebar.checkbox("IH (Industrial‑Heavy)", value=False, key="zone_ih")

min_addrpts = st.sidebar.number_input(
    "Min address points per parcel (multi‑tenant proxy)", value=0, min_value=0, step=1, key="min_addrpts"
)

st.sidebar.subheader("Scoring weights")
w_zoning = st.sidebar.slider("Zoning match", 0, 5, 3, key="w_zoning")
w_size_mid = st.sidebar.slider("Ideal size band (20k‑120k)", 0, 5, 2, key="w_size_mid")
w_addrpts = st.sidebar.slider("Address points ≥ 3", 0, 5, 2, key="w_addrpts")
w_depth = st.sidebar.slider("Shallow rectangle depth", 0, 3, 1, key="w_depth")
score_threshold = st.sidebar.slider("Candidate score threshold", 0, 10, 5, key="score_threshold")

# ---------------------------------
# Helpers
# ---------------------------------

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs=4326)
    return gdf.to_crs(epsg)


def min_rotated_rect_depth_ratio(poly: Polygon) -> Optional[float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) ** 0.5 for i in range(4)]
        edges.sort()
        short, long = edges[0], edges[2]
        return long / max(short, 1e-6)
    except Exception:
        return None


def allow_zoning_prefixes():
    allow = []
    if use_io: allow += ["IO"]
    if use_il: allow += ["IL"]
    if use_ic: allow += ["IC"]
    if use_im: allow += ["IM"]
    if use_ih: allow += ["IH"]
    return allow


def build_where_clause():
    # ZONING may contain letter+digit (e.g., IO1, IL2) — use LIKE on prefixes
    prefixes = allow_zoning_prefixes()
    if prefixes:
        likes = [f"ZONING LIKE '{p}%'" for p in prefixes]
        z_clause = "(" + " OR ".join(likes) + ")"
    else:
        z_clause = "1=1"
    sf_clause = f"BLDG_AREA >= {min_sf} AND BLDG_AREA <= {max_sf}"
    return f"{z_clause} AND {sf_clause}"

@st.cache_data(show_spinner=False)
def fetch_arcgis_geojson(layer_url: str, where: str, out_fields: str = "*", geometry: Optional[dict] = None) -> gpd.GeoDataFrame:
    """Fetch GeoJSON from an ArcGIS layer that supports geoJSON and return a GeoDataFrame (EPSG:4326)."""
    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }
    if geometry:
        params.update({
            "geometry": json.dumps(geometry),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })
    url = layer_url.rstrip('/') + "/query"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    gj = r.json()
    if not isinstance(gj, dict) or "features" not in gj:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(gj.get("features", []), crs="EPSG:4326")
    return gdf

# ---------------------------------
# Data loading — two modes
# ---------------------------------
parcels = None
addrpts = None

if mode == "Auto‑fetch (no downloads)":
    st.sidebar.subheader("Auto‑fetch options")
    extent_choice = st.sidebar.selectbox(
        "Area",
        ["Citywide (Industrial filter)", "Menomonee Valley demo", "Airport/College Ave demo"],
        index=0,
        key="extent_choice",
    )

    # Simple demo extents (lon/lat envelopes)
    extent = None
    if extent_choice == "Menomonee Valley demo":
        extent = {"xmin": -87.961, "ymin": 43.015, "xmax": -87.920, "ymax": 43.035}
    elif extent_choice == "Airport/College Ave demo":
        extent = {"xmin": -87.944, "ymin": 42.915, "xmax": -87.903, "ymax": 42.940}

    if st.sidebar.button("Fetch parcels", key="fetch_parcels_btn"):
        with st.spinner("Fetching parcels from City of Milwaukee…"):
            where = build_where_clause()
            out_fields = (
                "TAXKEY,OWNER_NAME_1,OWNER_MAIL_ADDR,OWNER_CITY_STATE,OWNER_ZIP,"
                "ZONING,LAND_USE,BLDG_AREA,UNIT,HOUSE_NR_LO,SDIR,STREET,STTYPE"
            )
            parcels = fetch_arcgis_geojson(PARCELS_LAYER, where, out_fields, geometry=extent)
        st.success(f"Loaded {len(parcels):,} parcels.")

    fetch_pts = st.sidebar.checkbox(
        "Also fetch Address Points for multi‑tenant signal", value=True, key="fetch_addrpts_chk"
    )
    if fetch_pts and parcels is not None and not parcels.empty:
        with st.spinner("Fetching Address Points…"):
            addrpts = fetch_arcgis_geojson(ADDRPTS_LAYER, "1=1", out_fields="*", geometry=extent)
        st.success(f"Loaded {len(addrpts):,} address points.")

else:
    st.sidebar.subheader("Upload files")
    parcels_file = st.sidebar.file_uploader(
        "Parcels (GeoPackage/GeoJSON/Shapefile)",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_parcels",
    )
    addrpts_file = st.sidebar.file_uploader(
        "(Optional) Address Points",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_addrpts",
    )
    bldg_file = st.sidebar.file_uploader(
        "(Optional) Building Footprints",
        type=["gpkg", "geojson", "json", "zip", "shp"],
        key="upload_bldgs",
    )

    def read_any_vector(file) -> gpd.GeoDataFrame:
        if file is None:
            return None
        return gpd.read_file(file)

    if parcels_file:
        parcels = read_any_vector(parcels_file)
    if addrpts_file:
        addrpts = read_any_vector(addrpts_file)
    if bldg_file:
        bldgs = read_any_vector(bldg_file)
        if bldgs is not None and not bldgs.empty and parcels is not None and not parcels.empty:
            bldgs = bldgs.to_crs(parcels.crs or "EPSG:4326")
            bldgs["foot_area"] = bldgs.geometry.area
            joined = gpd.sjoin(bldgs[["foot_area", "geometry"]], parcels[["geometry"]], how="inner", predicate="intersects")
            sums = joined.groupby(joined.index_right)["foot_area"].sum()
            parcels = parcels.copy()
            parcels["BLDG_AREA"] = sums

# Guard — need parcels to proceed
if parcels is None or len(parcels) == 0:
    st.info("⬅️ Use **Auto‑fetch** and click *Fetch parcels* (recommended), or upload a parcels file.")
    st.stop()

# ---------------------------------
# Feature engineering
# ---------------------------------
cols_lower = {c.lower(): c for c in parcels.columns}
getc = lambda *names: next((cols_lower[n] for n in names if n in cols_lower), None)

apn_col = getc("taxkey", "apn", "parcelid", "parcel_id")
owner1_col = getc("owner_name_1", "owner", "ownernme1")
mail_addr_col = getc("owner_mail_addr", "mailaddr", "owneraddr", "mailing_address")
mail_cityst_col = getc("owner_city_state", "mail_city_state")
mail_zip_col = getc("owner_zip", "mailzip")
zoning_col = getc("zoning")
landuse_col = getc("land_use", "landuse")
bldg_col = getc("bldg_area", "imprv_sqft", "bldg_sqft", "imprv_area")

# Fill missing BLDG_AREA with NaN so filters work
if bldg_col is None:
    parcels["BLDG_AREA"] = np.nan
    bldg_col = "BLDG_AREA"

# Address counts via spatial join
addr_count = pd.Series(0, index=parcels.index)
if addrpts is not None and not addrpts.empty:
    try:
        addrpts = addrpts.to_crs(parcels.crs or "EPSG:4326")
        parcels = parcels.to_crs(addrpts.crs)
        sjoin = gpd.sjoin(addrpts[["geometry"]], parcels[["geometry"]], how="inner", predicate="within")
        counts = sjoin.groupby(sjoin.index_right).size()
        addr_count = addr_count.add(counts, fill_value=0).astype(int)
    except Exception:
        pass

work = parcels.copy()
work["addr_count"] = addr_count

# Zoning flag (prefix check)
allow_prefixes = allow_zoning_prefixes()
if zoning_col is not None and allow_prefixes:
    ztxt = work[zoning_col].astype(str).str.upper().fillna("")
    work["zoning_ok"] = ztxt.str.startswith(tuple(allow_prefixes)) | ztxt.isin(allow_prefixes)
else:
    work["zoning_ok"] = True  # if no zoning, don’t exclude

# Size filter & depth ratio
work["bldg_sf"] = pd.to_numeric(work[bldg_col], errors="coerce")
work_3857 = ensure_crs(work)
work["depth_ratio"] = work_3857.geometry.apply(min_rotated_rect_depth_ratio)

# Scoring

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
    if row.get("depth_ratio") and row["depth_ratio"] > 1.2:
        score += w_depth
    return int(score)

work["score"] = work.apply(compute_score, axis=1)

# Effective min address points (auto‑relax if none loaded)
addrpts_loaded = addrpts is not None and len(addrpts) > 0
effective_min_addrpts = min_addrpts if addrpts_loaded else 0
if not addrpts_loaded:
    st.warning("No address points loaded — using Min address points = 0 for filtering.")

# Apply hard filters
mask = (
    work["bldg_sf"].between(min_sf, max_sf, inclusive="both") &
    work["zoning_ok"] &
    (work["addr_count"] >= effective_min_addrpts)
)

cand = work[mask].copy().sort_values("score", ascending=False)

if len(cand) == 0:
    st.error("No candidates yet. Try lowering Min building sf, enabling more zoning types (IC/IM/IH), or widen the area.")
else:
    st.markdown(f"**Candidates: {len(cand):,} parcels** meet your current filters.")

# ---------------------------------
# Map
# ---------------------------------
initial_view = pdk.ViewState(latitude=43.0389, longitude=-87.9065, zoom=11)

tile_layer = pdk.Layer("TileLayer", data=USGS_TILES, min_zoom=0, max_zoom=19, tile_size=256)

cand_geojson = json.loads(cand.to_crs(4326).to_json()) if len(cand) else {"type":"FeatureCollection","features":[]}
poly_layer = pdk.Layer(
    "GeoJsonLayer",
    data=cand_geojson,
    stroked=True,
    filled=True,
    get_fill_color=f"[score >= {score_threshold} ? 255 : 200, 100, 0, 60]",
    get_line_color=[255, 140, 0],
    get_line_width=1,
    pickable=True,
)

r = pdk.Deck(layers=[tile_layer, poly_layer], initial_view_state=initial_view, map_provider=None)
st.pydeck_chart(r, use_container_width=True)

st.caption("Imagery: USGS The National Map (USGSImageryOnly). Parcels & Address Points: City of Milwaukee MapServer (GeoJSON).")

# ---------------------------------
# Export call sheet
# ---------------------------------
export_cols = {
    "APN": apn_col,
    "Zoning": zoning_col,
    "Building SF": "bldg_sf",
    "Address Count": "addr_count",
    "Depth Ratio": "depth_ratio",
    "Score": "score",
    "Owner Name": owner1_col,
    "Owner Mailing Address": mail_addr_col,
    "Owner City/State": mail_cityst_col,
    "Owner Zip": mail_zip_col,
}

# Compose a property address if street fields exist
cols_lower = {c.lower(): c for c in work.columns}
parts = [cols_lower.get(p) for p in ["house_nr_lo", "sdir", "street", "sttype", "unit"] if cols_lower.get(p)]
if parts:
    addr_series = (
        work.get(cols_lower.get("house_nr_lo"), pd.Series("")).astype(str).str.replace(".0$", "", regex=True) + " " +
        work.get(cols_lower.get("sdir"), pd.Series("")).astype(str) + " " +
        work.get(cols_lower.get("street"), pd.Series("")).astype(str) + " " +
        work.get(cols_lower.get("sttype"), pd.Series("")).astype(str) +
        (" #" + work.get(cols_lower.get("unit"), pd.Series("")).astype(str)).where(work.get(cols_lower.get("unit"), pd.Series("")).astype(str).ne(""), "")
    ).str.replace(" +", " ", regex=True).str.strip()
    work["Property Address"] = addr_series
    export_cols = {"Property Address": "Property Address", **export_cols}

final_cols = [v for v in export_cols.values() if v is not None and v in work.columns]
export_df = work.loc[cand.index, final_cols].copy()
export_df.columns = [k for k, v in export_cols.items() if v is not None and v in work.columns]

st.subheader("Export call sheet")
st.dataframe(export_df.head(50))

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="milwaukee_small_bay_calls.csv", mime="text/csv", key="dl_csv")

# ---------------------------------
# Diagnostics
# ---------------------------------
with st.expander("Diagnostics"):
    st.write({
        "parcels_loaded": 0 if parcels is None else len(parcels),
        "addrpts_loaded": 0 if addrpts is None else len(addrpts),
        "candidates": len(cand),
        "filters": {
            "min_sf": min_sf,
            "max_sf": max_sf,
            "zoning_prefixes": allow_zoning_prefixes(),
            "min_addrpts_effective": effective_min_addrpts,
        },
    })
