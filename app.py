# app.py — Milwaukee Small‑Bay Industrial Finder (MVP)
# ---------------------------------------------------
# New: **No‑download mode** for Milwaukee — the app can auto‑fetch parcels and address points
# straight from the City of Milwaukee ArcGIS service (geoJSON supported). If you prefer, you can
# still upload local files. Imagery is USGS (public domain).
#
# Run locally
#   $ pip install streamlit geopandas shapely pydeck pandas numpy requests pyproj rtree
#   (optional, avoids GDAL headaches): pip install pyogrio && set GeoPandas to use it
#   $ streamlit run app.py
#
# Data sources used in auto‑fetch mode (public services):
#   Parcels (MPROP_full):
#     https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2
#     (Supports geoJSON queries; fields include TAXKEY, OWNER_*, ZONING, LAND_USE, BLDG_AREA.)
#   Address Points:
#     https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22
#     (Supports geoJSON; points used to estimate multi‑tenant via address counts.)
#   Imagery: USGS ImageryOnly tile service (public domain)

import io
import json
import math
import urllib.parse
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit as st
import pydeck as pdk
import requests

# Optional: prefer pyogrio if installed
try:
    gpd.options.io_engine = "pyogrio"
except Exception:
    pass

st.set_page_config(page_title="Milwaukee Small‑Bay Finder", layout="wide")
st.title("Milwaukee Small‑Bay Industrial Finder — MVP")

# ----------------------------
# Sidebar — choose data source
# ----------------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio(
    "Pick a mode",
    ["Auto‑fetch (no downloads)", "Upload files"],
    index=0,
)

# ----------------------------
# Sidebar — filters & scoring
# ----------------------------
st.sidebar.header("Filters & scoring")
min_sf = st.sidebar.number_input("Min building sf", value=10000, step=1000)
max_sf = st.sidebar.number_input("Max building sf", value=200000, step=1000)

st.sidebar.markdown("**Industrial zoning to include (Milwaukee 295‑801):**")
use_io = st.sidebar.checkbox("IO (Industrial‑Office)", value=True)
use_il = st.sidebar.checkbox("IL (Industrial‑Light)", value=True)
use_ic = st.sidebar.checkbox("IC (Industrial‑Commercial)", value=True)
use_im = st.sidebar.checkbox("IM (Industrial‑Mixed)", value=False)
use_ih = st.sidebar.checkbox("IH (Industrial‑Heavy)", value=False)

min_addrpts = st.sidebar.number_input("Min address points per parcel (multi‑tenant proxy)", value=2, min_value=0, step=1)

# scoring weights
st.sidebar.subheader("Scoring weights")
w_zoning = st.sidebar.slider("Zoning match", 0, 5, 3)
w_size_mid = st.sidebar.slider("Ideal size band (20k‑120k)", 0, 5, 2)
w_addrpts = st.sidebar.slider("Address points ≥ 3", 0, 5, 2)
w_depth = st.sidebar.slider("Shallow rectangle depth ratio", 0, 3, 1)
score_threshold = st.sidebar.slider("Candidate score threshold", 0, 10, 5)

# ----------------------------
# Helpers
# ----------------------------

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs=4326)
    return gdf.to_crs(epsg)


def min_rotated_rect_depth_ratio(poly) -> Optional[float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) ** 0.5 for i in range(4)]
        edges.sort()
        short, long = edges[0], edges[2]
        return long / max(short, 1e-6)
    except Exception:
        return None


def zone_filter_str_list():
    allow = []
    if use_io: allow += ["IO%"]
    if use_il: allow += ["IL%"]
    if use_ic: allow += ["IC"]
    if use_im: allow += ["IM"]
    if use_ih: allow += ["IH"]
    return allow


def build_where_clause():
    likes = []
    for z in zone_filter_str_list():
        if z.endswith('%'):
            likes.append(f"ZONING LIKE '{z}'")
        else:
            likes.append(f"ZONING = '{z}'")
    z_clause = "(" + " OR ".join(likes) + ")" if likes else "1=1"
    sf_clause = f"BLDG_AREA >= {min_sf} AND BLDG_AREA <= {max_sf}"
    return f"{z_clause} AND {sf_clause}"

@st.cache_data(show_spinner=False)
def fetch_arcgis_geojson(layer_url: str, where: str, out_fields: str, geometry: Optional[dict] = None) -> gpd.GeoDataFrame:
    """Fetch GeoJSON from an ArcGIS Feature/MapServer layer that supports geoJSON and return a GeoDataFrame (EPSG:4326)."""
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
    if "features" not in gj or not gj["features"]:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[] , crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    return gdf

# ----------------------------
# Load data — two modes
# ----------------------------
parcels = None
addrpts = None

PARCELS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2"
ADDRPTS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22"

if mode == "Auto‑fetch (no downloads)":
    st.sidebar.subheader("Auto‑fetch options")
    extent_choice = st.sidebar.selectbox(
        "Area",
        ["Citywide (Industrial filter)", "Menomonee Valley demo", "Airport/College Ave demo"],
        index=0,
        help="Use demos if you want a tiny subset to test instantly.",
    )

    # Simple demo extents (approximate lon/lat envelopes)
    extent = None
    if extent_choice == "Menomonee Valley demo":
        extent = {"xmin": -87.961, "ymin": 43.015, "xmax": -87.920, "ymax": 43.035}
    elif extent_choice == "Airport/College Ave demo":
        extent = {"xmin": -87.944, "ymin": 42.915, "xmax": -87.903, "ymax": 42.940}

    if st.sidebar.button("Fetch parcels"):
        with st.spinner("Fetching parcels from City of Milwaukee…"):
            where = build_where_clause()
            out_fields = "TAXKEY,OWNER_NAME_1,OWNER_MAIL_ADDR,OWNER_CITY_STATE,OWNER_ZIP,ZONING,LAND_USE,BLDG_AREA,UNIT"
            parcels = fetch_arcgis_geojson(PARCELS_LAYER, where, out_fields, geometry=extent)

        st.success(f"Loaded {len(parcels):,} parcels.")

    if st.sidebar.checkbox("Also fetch Address Points for multi‑tenant signal", value=True):
        if parcels is not None and not parcels.empty:
            with st.spinner("Fetching Address Points…"):
                # Filter address points to the same extent if provided to keep it light
                addrpts = fetch_arcgis_geojson(ADDRPTS_LAYER, "1=1", "StreetName,HouseNbr,Unit", geometry=extent)
            st.success(f"Loaded {len(addrpts):,} address points.")

else:
    # Upload mode
    st.sidebar.header("Upload files")
    parcels_file = st.sidebar.file_uploader("Parcels (GeoPackage/GeoJSON)", type=["gpkg", "geojson", "json", "zip", "shp"], help="Use GeoJSON or GeoPackage for easiest uploads.")
    addrpts_file = st.sidebar.file_uploader("(Optional) Address Points", type=["gpkg", "geojson", "json", "zip", "shp"]) 
    bldg_file = st.sidebar.file_uploader("(Optional) Building Footprints", type=["gpkg", "geojson", "json", "zip", "shp"]) 

    def read_any_vector(file) -> gpd.GeoDataFrame:
        if file is None:
            return None
        name = file.name.lower()
        if name.endswith(".gpkg") or name.endswith(".geojson") or name.endswith(".json"):
            return gpd.read_file(file)
        if name.endswith(".zip") or name.endswith(".shp"):
            return gpd.read_file(file)
        raise ValueError("Unsupported format.")

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

# Stop early if no parcels yet
if parcels is None or len(parcels) == 0:
    st.info("⬅️ Use **Auto‑fetch** and click *Fetch parcels* (recommended), or upload a parcels file.")
    st.stop()

# ----------------------------
# Feature engineering
# ----------------------------
# Normalize columns
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

# If missing building area, fill with 0 so filters work
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

# Zoning flag
allow_z = []
if use_io: allow_z += ["IO1", "IO2"]
if use_il: allow_z += ["IL1", "IL2"]
if use_ic: allow_z += ["IC"]
if use_im: allow_z += ["IM"]
if use_ih: allow_z += ["IH"]

if zoning_col is not None:
    ztxt = work[zoning_col].astype(str).str.upper()
    work["zoning_ok"] = False
    work.loc[ztxt.str.startswith(tuple([z[:2] for z in allow_z if z.endswith(('1','2'))])) | ztxt.isin(allow_z), "zoning_ok"] = True
else:
    work["zoning_ok"] = True  # if no zoning column, don’t exclude

# Size filter
work["bldg_sf"] = pd.to_numeric(work[bldg_col], errors="coerce")

# Depth ratio proxy (in projected meters/feet). Project to Web Mercator to measure.
work_3857 = ensure_crs(work)
work["depth_ratio"] = work_3857.geometry.apply(min_rotated_rect_depth_ratio)

# Score
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

# Apply hard filters
mask = (
    work["bldg_sf"].between(min_sf, max_sf, inclusive="both") &
    work["zoning_ok"] &
    (work["addr_count"] >= min_addrpts)
)

cand = work[mask].copy().sort_values("score", ascending=False)

st.markdown(f"**Candidates: {len(cand):,} parcels** meet your current filters.")

# ----------------------------
# Map
# ----------------------------
initial_view = pdk.ViewState(latitude=43.0389, longitude=-87.9065, zoom=11)

usgs_tiles = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

tile_layer = pdk.Layer("TileLayer", data=usgs_tiles, min_zoom=0, max_zoom=19, tile_size=256)

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
st.pydeck_chart(r)

st.caption("Imagery: USGS The National Map (USGSImageryOnly). Parcels & address points: City of Milwaukee MapServer (geoJSON queries).")

# ----------------------------
# Export call sheet
# ----------------------------
export_cols = {
    "APN": apn_col,
    "Property Address": None,  # composed below if street fields exist
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

# Try composing a street address if fields exist
street_parts = [p for p in ["house_nr_lo", "sdir", "street", "sttype", "unit"] if p in cols_lower]
if street_parts:
    addr_series = (
        work[cols_lower.get("house_nr_lo")].fillna("").astype(str).str.replace(".0$","", regex=True) + " " +
        work[cols_lower.get("sdir")].fillna("").astype(str) + " " +
        work[cols_lower.get("street")].fillna("").astype(str) + " " +
        work[cols_lower.get("sttype")].fillna("").astype(str) +
        (" #" + work[cols_lower.get("unit")].astype(str)).where(work[cols_lower.get("unit")].notna(), "")
    ).str.replace(" +", " ", regex=True).str.strip()
    work["_site_addr"] = addr_series
    export_cols["Property Address"] = "_site_addr"

final_cols = [v for v in export_cols.values() if v is not None and v in work.columns]
export_df = work.loc[cand.index, final_cols].copy()
export_df.columns = [k for k, v in export_cols.items() if v is not None and v in work.columns]

st.subheader("Export call sheet")
st.dataframe(export_df.head(50))

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="milwaukee_small_bay_calls.csv", mime="text/csv")

with st.expander("Notes / Tips"):
    st.markdown(
        """
        **No‑download mode** hits the City of Milwaukee MapServer layers and requests **geoJSON** directly.
        The parcels layer explicitly supports geoJSON and includes **ZONING** and **BLDG_AREA** fields.

        Use the **demo extents** if you just want to sanity‑check the workflow with a tiny subset.

        Want to save review labels and call outcomes? Wire a Postgres DB and add a `verifications` table.
        """
    )
# app.py — Milwaukee Small‑Bay Industrial Finder (MVP)
# ---------------------------------------------------
# New: **No‑download mode** for Milwaukee — the app can auto‑fetch parcels and address points
# straight from the City of Milwaukee ArcGIS service (geoJSON supported). If you prefer, you can
# still upload local files. Imagery is USGS (public domain).
#
# Run locally
#   $ pip install streamlit geopandas shapely pydeck pandas numpy requests pyproj rtree
#   (optional, avoids GDAL headaches): pip install pyogrio && set GeoPandas to use it
#   $ streamlit run app.py
#
# Data sources used in auto‑fetch mode (public services):
#   Parcels (MPROP_full):
#     https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2
#     (Supports geoJSON queries; fields include TAXKEY, OWNER_*, ZONING, LAND_USE, BLDG_AREA.)
#   Address Points:
#     https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22
#     (Supports geoJSON; points used to estimate multi‑tenant via address counts.)
#   Imagery: USGS ImageryOnly tile service (public domain)

import io
import json
import math
import urllib.parse
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import streamlit as st
import pydeck as pdk
import requests

# Optional: prefer pyogrio if installed
try:
    gpd.options.io_engine = "pyogrio"
except Exception:
    pass

st.set_page_config(page_title="Milwaukee Small‑Bay Finder", layout="wide")
st.title("Milwaukee Small‑Bay Industrial Finder — MVP")

# ----------------------------
# Sidebar — choose data source
# ----------------------------
st.sidebar.header("Data source")
mode = st.sidebar.radio(
    "Pick a mode",
    ["Auto‑fetch (no downloads)", "Upload files"],
    index=0,
)

# ----------------------------
# Sidebar — filters & scoring
# ----------------------------
st.sidebar.header("Filters & scoring")
min_sf = st.sidebar.number_input("Min building sf", value=10000, step=1000)
max_sf = st.sidebar.number_input("Max building sf", value=200000, step=1000)

st.sidebar.markdown("**Industrial zoning to include (Milwaukee 295‑801):**")
use_io = st.sidebar.checkbox("IO (Industrial‑Office)", value=True)
use_il = st.sidebar.checkbox("IL (Industrial‑Light)", value=True)
use_ic = st.sidebar.checkbox("IC (Industrial‑Commercial)", value=True)
use_im = st.sidebar.checkbox("IM (Industrial‑Mixed)", value=False)
use_ih = st.sidebar.checkbox("IH (Industrial‑Heavy)", value=False)

min_addrpts = st.sidebar.number_input("Min address points per parcel (multi‑tenant proxy)", value=2, min_value=0, step=1)

# scoring weights
st.sidebar.subheader("Scoring weights")
w_zoning = st.sidebar.slider("Zoning match", 0, 5, 3)
w_size_mid = st.sidebar.slider("Ideal size band (20k‑120k)", 0, 5, 2)
w_addrpts = st.sidebar.slider("Address points ≥ 3", 0, 5, 2)
w_depth = st.sidebar.slider("Shallow rectangle depth ratio", 0, 3, 1)
score_threshold = st.sidebar.slider("Candidate score threshold", 0, 10, 5)

# ----------------------------
# Helpers
# ----------------------------

def ensure_crs(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf is None:
        return None
    if gdf.crs is None:
        gdf = gpd.GeoDataFrame(gdf, geometry=gdf.geometry, crs=4326)
    return gdf.to_crs(epsg)


def min_rotated_rect_depth_ratio(poly) -> Optional[float]:
    try:
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        edges = [((coords[i][0]-coords[i+1][0])**2 + (coords[i][1]-coords[i+1][1])**2) ** 0.5 for i in range(4)]
        edges.sort()
        short, long = edges[0], edges[2]
        return long / max(short, 1e-6)
    except Exception:
        return None


def zone_filter_str_list():
    allow = []
    if use_io: allow += ["IO%"]
    if use_il: allow += ["IL%"]
    if use_ic: allow += ["IC"]
    if use_im: allow += ["IM"]
    if use_ih: allow += ["IH"]
    return allow


def build_where_clause():
    likes = []
    for z in zone_filter_str_list():
        if z.endswith('%'):
            likes.append(f"ZONING LIKE '{z}'")
        else:
            likes.append(f"ZONING = '{z}'")
    z_clause = "(" + " OR ".join(likes) + ")" if likes else "1=1"
    sf_clause = f"BLDG_AREA >= {min_sf} AND BLDG_AREA <= {max_sf}"
    return f"{z_clause} AND {sf_clause}"

@st.cache_data(show_spinner=False)
def fetch_arcgis_geojson(layer_url: str, where: str, out_fields: str, geometry: Optional[dict] = None) -> gpd.GeoDataFrame:
    """Fetch GeoJSON from an ArcGIS Feature/MapServer layer that supports geoJSON and return a GeoDataFrame (EPSG:4326)."""
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
    if "features" not in gj or not gj["features"]:
        return gpd.GeoDataFrame(columns=["geometry"], geometry=[] , crs="EPSG:4326")
    gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
    return gdf

# ----------------------------
# Load data — two modes
# ----------------------------
parcels = None
addrpts = None

PARCELS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/2"
ADDRPTS_LAYER = "https://milwaukeemaps.milwaukee.gov/arcgis/rest/services/property/parcels_mprop/MapServer/22"

if mode == "Auto‑fetch (no downloads)":
    st.sidebar.subheader("Auto‑fetch options")
    extent_choice = st.sidebar.selectbox(
        "Area",
        ["Citywide (Industrial filter)", "Menomonee Valley demo", "Airport/College Ave demo"],
        index=0,
        help="Use demos if you want a tiny subset to test instantly.",
    )

    # Simple demo extents (approximate lon/lat envelopes)
    extent = None
    if extent_choice == "Menomonee Valley demo":
        extent = {"xmin": -87.961, "ymin": 43.015, "xmax": -87.920, "ymax": 43.035}
    elif extent_choice == "Airport/College Ave demo":
        extent = {"xmin": -87.944, "ymin": 42.915, "xmax": -87.903, "ymax": 42.940}

    if st.sidebar.button("Fetch parcels"):
        with st.spinner("Fetching parcels from City of Milwaukee…"):
            where = build_where_clause()
            out_fields = "TAXKEY,OWNER_NAME_1,OWNER_MAIL_ADDR,OWNER_CITY_STATE,OWNER_ZIP,ZONING,LAND_USE,BLDG_AREA,UNIT"
            parcels = fetch_arcgis_geojson(PARCELS_LAYER, where, out_fields, geometry=extent)

        st.success(f"Loaded {len(parcels):,} parcels.")

    if st.sidebar.checkbox("Also fetch Address Points for multi‑tenant signal", value=True):
        if parcels is not None and not parcels.empty:
            with st.spinner("Fetching Address Points…"):
                # Filter address points to the same extent if provided to keep it light
                addrpts = fetch_arcgis_geojson(ADDRPTS_LAYER, "1=1", "StreetName,HouseNbr,Unit", geometry=extent)
            st.success(f"Loaded {len(addrpts):,} address points.")

else:
    # Upload mode
    st.sidebar.header("Upload files")
    parcels_file = st.sidebar.file_uploader("Parcels (GeoPackage/GeoJSON)", type=["gpkg", "geojson", "json", "zip", "shp"], help="Use GeoJSON or GeoPackage for easiest uploads.")
    addrpts_file = st.sidebar.file_uploader("(Optional) Address Points", type=["gpkg", "geojson", "json", "zip", "shp"]) 
    bldg_file = st.sidebar.file_uploader("(Optional) Building Footprints", type=["gpkg", "geojson", "json", "zip", "shp"]) 

    def read_any_vector(file) -> gpd.GeoDataFrame:
        if file is None:
            return None
        name = file.name.lower()
        if name.endswith(".gpkg") or name.endswith(".geojson") or name.endswith(".json"):
            return gpd.read_file(file)
        if name.endswith(".zip") or name.endswith(".shp"):
            return gpd.read_file(file)
        raise ValueError("Unsupported format.")

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

# Stop early if no parcels yet
if parcels is None or len(parcels) == 0:
    st.info("⬅️ Use **Auto‑fetch** and click *Fetch parcels* (recommended), or upload a parcels file.")
    st.stop()

# ----------------------------
# Feature engineering
# ----------------------------
# Normalize columns
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

# If missing building area, fill with 0 so filters work
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

# Zoning flag
allow_z = []
if use_io: allow_z += ["IO1", "IO2"]
if use_il: allow_z += ["IL1", "IL2"]
if use_ic: allow_z += ["IC"]
if use_im: allow_z += ["IM"]
if use_ih: allow_z += ["IH"]

if zoning_col is not None:
    ztxt = work[zoning_col].astype(str).str.upper()
    work["zoning_ok"] = False
    work.loc[ztxt.str.startswith(tuple([z[:2] for z in allow_z if z.endswith(('1','2'))])) | ztxt.isin(allow_z), "zoning_ok"] = True
else:
    work["zoning_ok"] = True  # if no zoning column, don’t exclude

# Size filter
work["bldg_sf"] = pd.to_numeric(work[bldg_col], errors="coerce")

# Depth ratio proxy (in projected meters/feet). Project to Web Mercator to measure.
work_3857 = ensure_crs(work)
work["depth_ratio"] = work_3857.geometry.apply(min_rotated_rect_depth_ratio)

# Score
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

# Apply hard filters
mask = (
    work["bldg_sf"].between(min_sf, max_sf, inclusive="both") &
    work["zoning_ok"] &
    (work["addr_count"] >= min_addrpts)
)

cand = work[mask].copy().sort_values("score", ascending=False)

st.markdown(f"**Candidates: {len(cand):,} parcels** meet your current filters.")

# ----------------------------
# Map
# ----------------------------
initial_view = pdk.ViewState(latitude=43.0389, longitude=-87.9065, zoom=11)

usgs_tiles = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"

tile_layer = pdk.Layer("TileLayer", data=usgs_tiles, min_zoom=0, max_zoom=19, tile_size=256)

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
st.pydeck_chart(r)

st.caption("Imagery: USGS The National Map (USGSImageryOnly). Parcels & address points: City of Milwaukee MapServer (geoJSON queries).")

# ----------------------------
# Export call sheet
# ----------------------------
export_cols = {
    "APN": apn_col,
    "Property Address": None,  # composed below if street fields exist
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

# Try composing a street address if fields exist
street_parts = [p for p in ["house_nr_lo", "sdir", "street", "sttype", "unit"] if p in cols_lower]
if street_parts:
    addr_series = (
        work[cols_lower.get("house_nr_lo")].fillna("").astype(str).str.replace(".0$","", regex=True) + " " +
        work[cols_lower.get("sdir")].fillna("").astype(str) + " " +
        work[cols_lower.get("street")].fillna("").astype(str) + " " +
        work[cols_lower.get("sttype")].fillna("").astype(str) +
        (" #" + work[cols_lower.get("unit")].astype(str)).where(work[cols_lower.get("unit")].notna(), "")
    ).str.replace(" +", " ", regex=True).str.strip()
    work["_site_addr"] = addr_series
    export_cols["Property Address"] = "_site_addr"

final_cols = [v for v in export_cols.values() if v is not None and v in work.columns]
export_df = work.loc[cand.index, final_cols].copy()
export_df.columns = [k for k, v in export_cols.items() if v is not None and v in work.columns]

st.subheader("Export call sheet")
st.dataframe(export_df.head(50))

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="milwaukee_small_bay_calls.csv", mime="text/csv")

with st.expander("Notes / Tips"):
    st.markdown(
        """
        **No‑download mode** hits the City of Milwaukee MapServer layers and requests **geoJSON** directly.
        The parcels layer explicitly supports geoJSON and includes **ZONING** and **BLDG_AREA** fields.

        Use the **demo extents** if you just want to sanity‑check the workflow with a tiny subset.

        Want to save review labels and call outcomes? Wire a Postgres DB and add a `verifications` table.
        """
    )
