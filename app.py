import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MiniMap, Fullscreen
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tempfile
from branca.colormap import LinearColormap
from folium import MacroElement
from jinja2 import Template

# --------------------------------------------------
# Configuración de la app
# --------------------------------------------------
st.set_page_config(page_title="GeoVisualizador de Transporte Público - Viena", layout="wide")

# --------------------------------------------------
# Rutas de datos (según tu Explorer)
# --------------------------------------------------
PATH_HALTE = "data/OeffHaltetest_clp.shp"
PATH_LINIEN = "data/Oefflinien_clp.shp"
PATH_BEZ   = "data/Bezirkgrenze.shp"
PATH_DEM   = "data/dem_wien.tif"

# Nombres de columnas
BEZ_NAME_COL = "NAMEG"        # Nombre del distrito
BEZ_CODE_COL = "BEZ"          # Código del distrito
HAL_NAME_COL = "HTXT"         # Nombre de la parada
HAL_LINES_COL = "HLINIEN"     # Líneas que paran
HAL_WEB_COL = "WEBLINK1"      # Enlace
LIN_NAME_COL = "LBEZEICHNU"   # Nombre de la línea
LIN_TYPE_COL = "LTYPTXT"      # Tipo (Tranvía, S‑Bahn, ...)

# --------------------------------------------------
# Carga de datos
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    gdf_hal = gpd.read_file(PATH_HALTE)
    gdf_lin = gpd.read_file(PATH_LINIEN)
    gdf_bez = gpd.read_file(PATH_BEZ)

    # Asegurar WGS84
    if not gdf_hal.crs or gdf_hal.crs.to_epsg() != 4326:
        gdf_hal = gdf_hal.to_crs(4326)
    if not gdf_lin.crs or gdf_lin.crs.to_epsg() != 4326:
        gdf_lin = gdf_lin.to_crs(4326)
    if not gdf_bez.crs or gdf_bez.crs.to_epsg() != 4326:
        gdf_bez = gdf_bez.to_crs(4326)

    # Columnas relevantes
    keep_hal = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal.columns] + ["geometry"]
    keep_lin = [c for c in [LIN_NAME_COL, LIN_TYPE_COL] if c in gdf_lin.columns] + ["geometry"]
    keep_bez = [c for c in [BEZ_NAME_COL, BEZ_CODE_COL] if c in gdf_bez.columns] + ["geometry"]
    gdf_hal = gdf_hal[keep_hal].copy()
    gdf_lin = gdf_lin[keep_lin].copy()
    gdf_bez = gdf_bez[keep_bez].copy()

    return gdf_hal, gdf_lin, gdf_bez

gdf_hal, gdf_lin, gdf_bez = load_data()

# --------------------------------------------------
# Auxiliares
# --------------------------------------------------
def lista_distritos(gdf_bez):
    vals = sorted(gdf_bez[BEZ_NAME_COL].astype(str).dropna().unique().tolist()) if BEZ_NAME_COL in gdf_bez.columns else []
    return ["Todos"] + vals

def filtrar_por_distrito(gdf_pts, gdf_lin, gdf_bez, sel_name):
    if sel_name == "Todos":
        return gdf_pts, gdf_lin, None
    if BEZ_NAME_COL not in gdf_bez.columns:
        return gdf_pts.iloc[0:0], gdf_lin.iloc[0:0], None
    poly = gdf_bez[gdf_bez[BEZ_NAME_COL] == sel_name]
    if poly.empty:
        return gdf_pts.iloc[0:0], gdf_lin.iloc[0:0], None
    pts_sel = gpd.sjoin(gdf_pts, poly[["geometry"]], predicate="within", how="inner").drop(columns=["index_right"])
    lin_sel = gpd.sjoin(gdf_lin, poly[["geometry"]], predicate="intersects", how="inner").drop(columns=["index_right"])
    return pts_sel, lin_sel, poly

def area_km2(poly_gdf):
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
                name="MDE",
                image=tmp.name,
                bounds=[[bottom, left], [top, right]],
                opacity=1.0,
                interactive=False,
                cross_origin=False
            ).add_to(map_obj)

            colors = [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, 256)]
            colormap = LinearColormap(colors, vmin=vmin, vmax=vmax)
            colormap.caption = "Elevación (m)"
            colormap.add_to(map_obj)
    except Exception as e:
        st.warning(f"No se pudo cargar el MDE: {e}")

# --------------------------------------------------
# Sidebar (español)
# --------------------------------------------------
st.sidebar.header("Capas")
show_hal = st.sidebar.checkbox("Paradas", True)
show_lin = st.sidebar.checkbox("Líneas", True)
show_bez = st.sidebar.checkbox("Distritos", True)
show_dem = st.sidebar.checkbox("MDE", False)

st.sidebar.markdown("---")
basemap = st.sidebar.radio("Mapa base", ["OpenStreetMap", "Satélite"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Seleccionar distrito")
opciones_distrito = lista_distritos(gdf_bez)
sel_bez = st.sidebar.selectbox("Distrito", opciones_distrito, index=0)

# Datos filtrados
gdf_hal_f, gdf_lin_f, gdf_bez_sel = filtrar_por_distrito(gdf_hal, gdf_lin, gdf_bez, sel_bez)

# --------------------------------------------------
# Mapa
# --------------------------------------------------
center = [48.2082, 16.3738]  # Viena
m = folium.Map(location=center, zoom_start=12, tiles=None)

# Mapas base
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=(basemap == "OpenStreetMap")).add_to(m)
folium.TileLayer("Esri.WorldImagery", name="Satélite", show=(basemap == "Satélite")).add_to(m)

# MDE
if show_dem:
    add_dem_overlay(m, PATH_DEM, cmap_name="terrain", alpha=0.55)

# Distritos
if show_bez and not gdf_bez.empty:
    if sel_bez == "Todos" or gdf_bez_sel is None:
        bez_draw = gdf_bez
        style = lambda f: {"fillColor": "#00000000", "color": "#555", "weight": 1.2}
    else:
        bez_draw = gdf_bez_sel
        style = lambda f: {"fillColor": "#ff990055", "color": "#ff9900", "weight": 2}
    tooltip_fields = [BEZ_NAME_COL] if BEZ_NAME_COL in gdf_bez.columns else []
    folium.GeoJson(
        bez_draw[[*(tooltip_fields), "geometry"]],
        name="Distritos",
        style_function=style,
        tooltip=folium.features.GeoJsonTooltip(fields=tooltip_fields, aliases=["Distrito:"], sticky=False) if tooltip_fields else None
    ).add_to(m)

# Líneas
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
        name="Líneas",
        style_function=style_line,
        tooltip=folium.features.GeoJsonTooltip(fields=fields, aliases=["Línea:", "Tipo:"], sticky=False) if fields else None
    ).add_to(m)

# Leyenda categórica visible para líneas
legend_html = """
<div style="
  position: fixed;
  bottom: 90px; left: 10px;
  z-index: 9999;
  background-color: white;
  padding: 8px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 13px;">
  <b>Leyenda – Líneas</b><br>
  <span style="background:#d7191c;width:12px;height:12px;display:inline-block;margin-right:6px;"></span> Tranvía<br>
  <span style="background:#2c7bb6;width:12px;height:12px;display:inline-block;margin-right:6px;"></span> S‑Bahn<br>
  <span style="background:#fdae61;width:12px;height:12px;display:inline-block;margin-right:6px;"></span> Bus regional<br>
  <span style="background:#1a9641;width:12px;height:12px;display:inline-block;margin-right:6px;"></span> Bus urbano<br>
  <span style="background:#984ea3;width:12px;height:12px;display:inline-block;margin-right:6px;"></span> Tren
</div>
"""
class Legend(MacroElement):
    def __init__(self, html):
        super().__init__()
        self._template = Template(f"""{{% macro script(this, kwargs) %}}
            var legend = $(`{legend_html}`);
            $('body').append(legend);
        {{% endmacro %}}""")
Legend(legend_html).add_to(m)

# Paradas
if show_hal and not gdf_hal_f.empty:
    use_cols = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal_f.columns]
    draw_df = gdf_hal_f[use_cols + ["geometry"]] if use_cols else gdf_hal_f
    for _, r in draw_df.iterrows():
        lat, lon = r.geometry.y, r.geometry.x
        parts = []
        if HAL_NAME_COL in r: parts.append(f"Parada: {r[HAL_NAME_COL]}")
        if HAL_LINES_COL in r: parts.append(f"Líneas: {r[HAL_LINES_COL]}")
        tooltip = "<br>".join(parts) if parts else None
        popup = f'<a href="{r[HAL_WEB_COL]}" target="_blank">Horario</a>' if (HAL_WEB_COL in r and pd.notna(r[HAL_WEB_COL])) else None
        folium.CircleMarker(
            [lat, lon], radius=2, color="#004b87", fill=True, fill_color="#2b83ba",
            fill_opacity=0.8, weight=0.3, tooltip=tooltip, popup=popup
        ).add_to(m)

# Plugins + control de capas
MiniMap(toggle_display=True).add_to(m)
Fullscreen(position="topleft").add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# --------------------------------------------------
# Encabezado + Introducción
# --------------------------------------------------
st.markdown("### GeoVisualizador de Accesibilidad al Transporte Público en Viena")
st.markdown(
    "Explora la accesibilidad al transporte público en Viena. "
    "En la barra lateral puedes activar/ocultar capas (paradas, líneas, distritos y MDE), "
    "cambiar el mapa base y filtrar por distrito. En la parte inferior encontrarás estadísticas, "
    "un gráfico con el número de paradas por distrito y una tabla de atributos."
)

out = st_folium(m, width="100%", height=650)

# --------------------------------------------------
# Estadísticas (sidebar)
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Estadísticas")
anz_hal = len(gdf_hal_f)
anz_lin_unique = gdf_lin_f[LIN_NAME_COL].nunique() if LIN_NAME_COL in gdf_lin_f.columns and not gdf_lin_f.empty else len(gdf_lin_f)
area = area_km2(gdf_bez_sel)
st.sidebar.metric("Paradas", f"{anz_hal}")
st.sidebar.metric("Líneas (únicas)", f"{anz_lin_unique}")
st.sidebar.metric("Área", f"{area:.2f} km²" if area else "—")

# --------------------------------------------------
# Gráfico: Paradas por distrito
# --------------------------------------------------
st.markdown("### Paradas por distrito")
try:
    if BEZ_NAME_COL in gdf_bez.columns:
        joined = gpd.sjoin(gdf_hal, gdf_bez[[BEZ_NAME_COL, "geometry"]], predicate="within", how="inner")
        counts = joined.groupby(BEZ_NAME_COL).size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4))
        counts.plot(kind="bar", ax=ax, color="#2b83ba")
        ax.set_ylabel("Número de paradas")
        ax.set_xlabel("Distrito")
        ax.set_title("Paradas por distrito")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No se encontró la columna de nombre de distrito – se omite el gráfico.")
except Exception as e:
    st.info(f"No se pudo generar el gráfico: {e}")

# --------------------------------------------------
# Tabla de atributos
# --------------------------------------------------
st.markdown("### Tabla de atributos – Paradas (filtradas)")
table_cols = [c for c in [HAL_NAME_COL, HAL_LINES_COL, HAL_WEB_COL] if c in gdf_hal_f.columns]
if table_cols:
    st.dataframe(
        gdf_hal_f[table_cols].rename(columns={
            HAL_NAME_COL: "Parada",
            HAL_LINES_COL: "Líneas",
            HAL_WEB_COL: "Enlace"
        })
    )
else:
    st.write("No se encontraron atributos para mostrar.")
