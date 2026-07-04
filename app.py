import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from branca.colormap import linear

# --------------------------------------------------
# App-Konfiguration
# --------------------------------------------------
st.set_page_config(page_title="GeoVisualizador Transporte Público - Viena", layout="wide")

# --------------------------------------------------
# Pfade (an deine Dateien angepasst)
# --------------------------------------------------
PATH_HALTE = "data/OeffHaltestest_clp.shp"
PATH_LINIEN = "data/Oefflinien_clp.shp"
PATH_BEZ   = "data/Bezirksgrenze.shp"
PATH_DEM   = "data/dem_wien.tif"

# Spaltennamen
BEZ_NAME_COL = "NAMEG"        # Bezirksname (DÖBLING, ...)
BEZ_CODE_COL = "BEZ"          # 01, 02, ...
HAL_NAME_COL = "HTXT"         # Haltestellenname
HAL_LINES_COL = "HLINIEN"     # bediente Linien
HAL_WEB_COL = "WEBLINK1"      # Link
LIN_NAME_COL = "LBEZEICHNU"   # Linienbezeichnung
LIN_TYPE_COL = "LTYPTXT"      # Typ (Straßenbahn, S‑Bahn, ...)

# --------------------------------------------------
# Daten laden
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    gdf_hal = gpd.read_file(PATH_HALTE)
    gdf_lin = gpd.read_file(PATH_LINIEN)
    gdf_bez = gpd.read_file(PATH_BEZ)

    # Sicherstellen: WGS84
    if not gdf_hal.crs or gdf_hal.crs.to_epsg() != 4326:
        gdf_hal = gdf_hal.to_crs(4326)
    if not gdf_lin.crs or gdf_lin.crs.to_epsg() != 4326:
        gdf_lin = gdf_lin.to_crs(4326)
    if not gdf_bez.crs or gdf_bez.crs.to_epsg() != 4326:
        gdf_bez = gdf_bez.to_crs(4326)

    # Relevante Spalten
    keep_hal = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal.columns] + ["geometry"]
    keep_lin = [c for c in [LIN_NAME_COL, LIN_TYPE_COL] if c in gdf_lin.columns] + ["geometry"]
    keep_bez = [c for c in [BEZ_NAME_COL, BEZ_CODE_COL] if c in gdf_bez.columns] + ["geometry"]
    gdf_hal = gdf_hal[keep_hal].copy()
    gdf_lin = gdf_lin[keep_lin].copy()
    gdf_bez = gdf_bez[keep_bez].copy()

    return gdf_hal, gdf_lin, gdf_bez

gdf_hal, gdf_lin, gdf_bez = load_data()

# --------------------------------------------------
# Helfer
# --------------------------------------------------
def bezirksliste(gdf_bez):
    vals = sorted(gdf_bez[BEZ_NAME_COL].astype(str).dropna().unique().tolist())
    return ["Alle"] + vals

def filter_by_bezirk(gdf_pts, gdf_lin, gdf_bez, sel_name):
    if sel_name == "Alle":
        return gdf_pts, gdf_lin, None
    poly = gdf_bez[gdf_bez[BEZ_NAME_COL] == sel_name]
    if poly.empty:
        return gdf_pts.iloc[0:0], gdf_lin.iloc[0:0], None
    pts_sel = gpd.sjoin(gdf_pts, poly[["geometry"]], predicate="within", how="inner").drop(columns=["index_right"])
    lin_sel = gpd.sjoin(gdf_lin, poly[["geometry"]], predicate="intersects", how="inner").drop(columns=["index_right"])
    return pts_sel, lin_sel, poly

def flaeche_km2(poly_gdf):
    if poly_gdf is None or poly_gdf.empty:
        return None
    poly_m = poly_gdf.to_crs(3857)
    return float(poly_m.geometry.area.sum())/1e6

def add_dem_overlay(map_obj, dem_path, cmap_name="terrain", alpha=0.55):
    try:
        with rasterio.open(dem_path) as src:
            arr = src.read(1).astype(float)
            mask = src.read_masks(1) == 0
            arr[mask] = np.nan
            if np.all(np.isnan(arr)):
                return
            vmin = np.nanpercentile(arr, 2)
            vmax = np.nanpercentile(arr, 98)
            norm = (arr - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)

            cmap = plt.get_cmap(cmap_name)
            rgba = cmap(norm)
            rgba[..., 3] = np.where(np.isnan(arr), 0, alpha)

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            plt.imsave(tmp.name, rgba)

            left, bottom, right, top = rasterio.transform.array_bounds(src.height, src.width, src.transform)

            folium.raster_layers.ImageOverlay(
                name="DEM",
                image=tmp.name,
                bounds=[[bottom, left], [top, right]],
                opacity=1.0,
                interactive=False,
                cross_origin=False
            ).add_to(map_obj)

            colormap = getattr(linear, cmap_name).scale(vmin, vmax)
            colormap.caption = "Höhe (m)"
            colormap.add_to(map_obj)
    except Exception as e:
        st.warning(f"DEM konnte nicht geladen werden: {e}")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Layer")
show_hal = st.sidebar.checkbox("Haltestellen", True)
show_lin = st.sidebar.checkbox("Linien", True)
show_bez = st.sidebar.checkbox("Bezirke", True)
show_dem = st.sidebar.checkbox("DEM", True)

st.sidebar.markdown("---")
basemap = st.sidebar.radio("Basemap", ["OpenStreetMap", "Satellite"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Bezirk auswählen")
bezirk_opt = bezirksliste(gdf_bez)
sel_bez = st.sidebar.selectbox("Bezirk", bezirk_opt, index=0)

# Gefilterte Daten
gdf_hal_f, gdf_lin_f, gdf_bez_sel = filter_by_bezirk(gdf_hal, gdf_lin, gdf_bez, sel_bez)

# --------------------------------------------------
# Karte
# --------------------------------------------------
center = [48.2082, 16.3738]  # Wien
m = folium.Map(location=center, zoom_start=12, tiles=None)

# Basemaps
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=(basemap == "OpenStreetMap")).add_to(m)
folium.TileLayer("Esri.WorldImagery", name="Satellite", show=(basemap == "Satellite")).add_to(m)

# DEM
if show_dem:
    add_dem_overlay(m, PATH_DEM, cmap_name="terrain", alpha=0.55)

# Bezirke
if show_bez:
    if sel_bez == "Alle":
        bez_draw = gdf_bez
        style = lambda f: {"fillColor": "#00000000", "color": "#555", "weight": 1.2}
    else:
        bez_draw = gdf_bez_sel
        style = lambda f: {"fillColor": "#ff990055", "color": "#ff9900", "weight": 2}
    folium.GeoJson(
        bez_draw[[BEZ_NAME_COL, "geometry"]],
        name="Bezirke",
        style_function=style,
        tooltip=folium.features.GeoJsonTooltip(fields=[BEZ_NAME_COL], aliases=["Bezirk:"], sticky=False)
    ).add_to(m)

# Linien
if show_lin and not gdf_lin_f.empty:
    type_colors = {
        "Straßenbahn": "#d7191c",
        "S-Bahn": "#2c7bb6",
        "Regionalbus": "#fdae61",
        "Stadtbus": "#1a9641",
        "Zug": "#984ea3"
    }
    def style_line(ft):
        t = ft["properties"].get(LIN_TYPE_COL, "")
        color = type_colors.get(t, "#555")
        weight = 4 if t in ["S-Bahn", "Zug"] else 3
        return {"color": color, "weight": weight}

    fields = [c for c in [LIN_NAME_COL, LIN_TYPE_COL] if c in gdf_lin_f.columns]
    folium.GeoJson(
        gdf_lin_f[fields + ["geometry"]] if fields else gdf_lin_f[["geometry"]],
        name="Linien",
        style_function=style_line,
        tooltip=folium.features.GeoJsonTooltip(fields=fields, aliases=["Linie:", "Typ:"], sticky=False) if fields else None
    ).add_to(m)

# Haltestellen
if show_hal and not gdf_hal_f.empty:
    use_cols = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal_f.columns]
    draw_df = gdf_hal_f[use_cols + ["geometry"]] if use_cols else gdf_hal_f
    for _, r in draw_df.iterrows():
        lat, lon = r.geometry.y, r.geometry.x
        parts = []
        if HAL_NAME_COL in r: parts.append(f"Haltestelle: {r[HAL_NAME_COL]}")
        if HAL_LINES_COL in r: parts.append(f"Linien: {r[HAL_LINES_COL]}")
        tooltip = "<br>".join(parts) if parts else None
        popup = f'<a href="{r[HAL_WEB_COL]}" target="_blank">Fahrplan</a>' if (HAL_WEB_COL in r and pd.notna(r[HAL_WEB_COL])) else None
        folium.CircleMarker(
            [lat, lon], radius=2, color="#004b87", fill=True, fill_color="#2b83ba",
            fill_opacity=0.8, weight=0.3, tooltip=tooltip, popup=popup
        ).add_to(m)

# Plugins + LayerControl
MiniMap(toggle_display=True).add_to(m)
Fullscreen(position="topleft").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st.markdown("### GeoVisualizador de Accesibilidad al Transporte Público en Viena")
out = st_folium(m, width="100%", height=650)

# --------------------------------------------------
# Statistik (Sidebar)
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Statistik")
anz_hal = len(gdf_hal_f)
anz_lin_unique = gdf_lin_f[LIN_NAME_COL].nunique() if LIN_NAME_COL in gdf_lin_f.columns and not gdf_lin_f.empty else len(gdf_lin_f)
area = flaeche_km2(gdf_bez_sel)
st.sidebar.metric("Haltestellen", f"{anz_hal}")
st.sidebar.metric("Linien (unique)", f"{anz_lin_unique}")
st.sidebar.metric("Fläche", f"{area:.2f} km²" if area else "—")

# --------------------------------------------------
# Diagramm: Haltestellen pro Bezirk
# --------------------------------------------------
st.markdown("### Haltestellen pro Bezirk")
try:
    joined = gpd.sjoin(gdf_hal, gdf_bez[[BEZ_NAME_COL, "geometry"]], predicate="within", how="inner")
    counts = joined.groupby(BEZ_NAME_COL).size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax, color="#2b83ba")
    ax.set_ylabel("Anzahl Haltestellen")
    ax.set_xlabel("Bezirk")
    ax.set_title("Haltestellen pro Bezirk")
    plt.tight_layout()
    st.pyplot(fig)
except Exception as e:
    st.info(f"Diagramm konnte nicht erzeugt werden: {e}")

# --------------------------------------------------
# Attributtabelle
# --------------------------------------------------
st.markdown("### Attributtabelle – Haltestellen (gefiltert)")
table_cols = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal_f.columns]
if table_cols:
    st.dataframe(
        gdf_hal_f[table_cols].rename(columns={
            HAL_NAME_COL: "Haltestelle",
            HAL_LINES_COL: "Linien",
            HAL_WEB_COL: "Link"
        })
    )
else:
    st.write("Keine anzeigbaren Attribute gefunden.")