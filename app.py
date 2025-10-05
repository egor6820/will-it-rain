# app.py
"""
NASA Weather Globe — improved UI (user-friendly summary + debug expander)
Replace your current app.py with this file.
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import datetime
import os
import math

# Optional mapping libraries
USE_FOLIUM = False
USE_STREAMLIT_FOLIUM = False
try:
    import folium
    from folium.plugins import MarkerCluster
    from streamlit_folium import st_folium
    USE_FOLIUM = True
    USE_STREAMLIT_FOLIUM = True
except Exception:
    USE_FOLIUM = False
    USE_STREAMLIT_FOLIUM = False

USE_PYDECK = False
try:
    import pydeck as pdk
    USE_PYDECK = True
except Exception:
    USE_PYDECK = False

# --------------------
# Config & constants
# --------------------
st.set_page_config(page_title="NASA Weather Globe", page_icon="🌍", layout="wide")
MAP_HEIGHT = 720

LOCAL_DATA_DIR = Path("data")
LOCAL_CITIES_FILE = LOCAL_DATA_DIR / "cities.json"

GIBS_LAYER = "MODIS_Terra_Land_Surface_Temp_Day"
NASA_GIBS_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/"
    "GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"
)

TEMP_MIN = -50
TEMP_MAX = 50
NEAREST_SUGGESTIONS = 8

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# --------------------
# i18n minimal
# --------------------
DEFAULT_LANG = "en"
I18N = {
    "en": {
        "title": "NASA Weather Globe — Hybrid",
        "search_placeholder": "Search by city or coords (e.g. Kyiv or 50.45,30.52)",
        "map_options": "Map options",
        "modis_date": "MODIS date",
        "show_modis": "Show MODIS LST (NASA GIBS)",
        "base_map": "Base map",
        "pending": "PENDING",
        "confirmed": "CONFIRMED",
        "confirm": "Confirm",
        "cancel": "Cancel",
        "nearby_quick": "Nearby (quick)",
        "download_csv": "Download hourly CSV",
        "next_48h": "Next 48h (preview)",
        "seven_day": "7-day",
        "no_weather": "Confirm a point to load weather details here.",
        "recent": "Recent selections",
        "nearest_label": "Nearest cities (choose one)",
        "coords_label": "Coordinates",
        "lang_toggle": "EN/UA",
        "loading_local": "Loading local city DB...",
        "local_missing": "Local cities.json not found — using small fallback.",
    }
}
lang = "en"

# --------------------
# Utilities
# --------------------
def temp_to_hex(t: Optional[float], vmin=TEMP_MIN, vmax=TEMP_MAX) -> str:
    if t is None:
        return "#999999"
    tv = max(min(t, vmax), vmin)
    ratio = (tv - vmin) / (vmax - vmin)
    if ratio < 0.5:
        g = int((ratio / 0.5) * 200)
        b = int(200 + (1 - ratio / 0.5) * 55)
        r = 0
    else:
        r = int(((ratio - 0.5) / 0.5) * 240)
        g = int(200 - ((ratio - 0.5) / 0.5) * 200)
        b = 0
    r = max(0, min(255, r)); g = max(0, min(255, g)); b = max(0, min(255, b))
    return f"#{r:02x}{g:02x}{b:02x}"

def make_legend_svg(vmin=TEMP_MIN, vmax=TEMP_MAX, width=320, height=56) -> str:
    steps = 200
    stops = []
    for i in range(steps + 1):
        ratio = i / steps
        val = vmin + ratio * (vmax - vmin)
        color = temp_to_hex(val, vmin, vmax)
        stops.append(f'<stop offset="{ratio*100}%" stop-color="{color}" />')
    svg = f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs><linearGradient id="g" x1="0" x2="1">{''.join(stops)}</linearGradient></defs>
      <rect x="0" y="0" width="{width}" height="{height-20}" fill="url(#g)" stroke="#000"/>
      <text x="6" y="{height-6}" font-size="12" fill="#000">{vmin}°C</text>
      <text x="{width-56}" y="{height-6}" font-size="12" fill="#000">{vmax}°C</text>
    </svg>
    '''
    return svg

def haversine_array(lat1, lon1, lat2_array, lon2_array):
    R = 6371.0
    lat1r = np.radians(lat1); lon1r = np.radians(lon1)
    lat2r = np.radians(np.array(lat2_array, dtype=float))
    lon2r = np.radians(np.array(lon2_array, dtype=float))
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --------------------
# Network helpers
# --------------------
@st.cache_data(ttl=60*60)
def nominatim_search(q: str, limit: int = 12):
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": q, "format": "json", "limit": limit},
                         headers={"User-Agent": "streamlit-nasa-globe"}, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []
    return []

@st.cache_data(ttl=60*60*3)
def get_weather_open_meteo(lat: float, lon: float, days: int = 7):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation_probability,wind_speed_10m,cloudcover,uv_index"
            f"&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset"
            f"&current_weather=true&forecast_days={days}&timezone=auto"
        )
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_data(ttl=60*60*24)
def get_altitude_opentopo(lat: float, lon: float):
    try:
        r = requests.get("https://api.opentopodata.org/v1/test-dataset", params={"locations": f"{lat},{lon}"}, timeout=8)
        if r.status_code == 200:
            j = r.json()
            if "results" in j and j["results"]:
                return j["results"][0].get("elevation")
    except Exception:
        return None
    return None

# --------------------
# Local city DB
# --------------------
SMALL_FALLBACK = [
    {"display_name":"Kyiv, UA", "lat":50.4501, "lon":30.5234},
    {"display_name":"Moscow, Russia", "lat":55.7558, "lon":37.6173},
    {"display_name":"Warsaw, Poland", "lat":52.2297, "lon":21.0122},
    {"display_name":"Berlin, Germany", "lat":52.52, "lon":13.405},
    {"display_name":"Paris, France", "lat":48.8566, "lon":2.3522},
    {"display_name":"Rome, Italy", "lat":41.9028, "lon":12.4964},
    {"display_name":"Istanbul, Turkey", "lat":41.0082, "lon":28.9784},
    {"display_name":"Budapest, Hungary", "lat":47.4979, "lon":19.0402},
    {"display_name":"Bucharest, Romania", "lat":44.4268, "lon":26.1025},
    {"display_name":"Prague, Czechia", "lat":50.0755, "lon":14.4378},
]

def load_local_cities():
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_CITIES_FILE.exists():
        try:
            with open(LOCAL_CITIES_FILE, "r", encoding="utf-8") as f:
                j = json.load(f)
            df = pd.DataFrame(j)
            if "lng" in df.columns and "lon" not in df.columns:
                df["lon"] = df["lng"]
            if "latitude" in df.columns and "lat" not in df.columns:
                df["lat"] = df["latitude"]
            if "longitude" in df.columns and "lon" not in df.columns:
                df["lon"] = df["longitude"]
            if "display_name" not in df.columns:
                if "name" in df.columns and "country" in df.columns:
                    df["display_name"] = df["name"].astype(str) + ", " + df["country"].astype(str)
                elif "name" in df.columns:
                    df["display_name"] = df["name"].astype(str)
                else:
                    df["display_name"] = df.index.astype(str)
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
            df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)
            df = df[["display_name","lat","lon"]]
            return df
        except Exception:
            st.warning(I18N[lang]["local_missing"])
            return pd.DataFrame(SMALL_FALLBACK)
    else:
        st.info(I18N[lang]["local_missing"])
        return pd.DataFrame(SMALL_FALLBACK)

# --------------------
# Session state init
# --------------------
if "confirmed" not in st.session_state:
    st.session_state.confirmed = {"lat": 50.4501, "lon": 30.5234, "city": "Kyiv, UA"}
if "pending" not in st.session_state:
    st.session_state.pending = None
if "weather" not in st.session_state:
    st.session_state.weather = None
if "nearby" not in st.session_state:
    st.session_state.nearby = []
if "mru" not in st.session_state:
    st.session_state.mru = []

# Load cities
with st.spinner(I18N[lang]["loading_local"]):
    cities_df = load_local_cities()
city_lats = cities_df["lat"].to_numpy()
city_lons = cities_df["lon"].to_numpy()
city_disp = cities_df["display_name"].to_numpy()

# --------------------
# Feature extraction & backend call
# --------------------
def extract_simple_features_from_weather(w: Dict[str,Any]):
    if not w:
        return None
    hourly = w.get("hourly", {})
    temps = hourly.get("temperature_2m", [])
    humid = hourly.get("relative_humidity_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precip_prob = hourly.get("precipitation_probability", [])
    uvs = hourly.get("uv_index", [])
    try:
        max_temp = float(max(temps)) if len(temps) else 0.0
        avg_hum = float(np.mean(humid)) if len(humid) else 0.0
        max_wind = float(max(winds)) if len(winds) else 0.0
        mean_precip = float(np.mean(precip_prob)) if len(precip_prob) else 0.0
        mean_uv = float(np.mean(uvs)) if len(uvs) else 0.0
        return {"temp": max_temp, "hum": avg_hum, "wind": max_wind, "precip": mean_precip, "uv": mean_uv}
    except Exception:
        return None

def call_backend_prediction(aggregates: Dict[str,Any]):
    if not aggregates:
        return None
    try:
        r = requests.post(f"{BACKEND_URL.rstrip('/')}/predict", json=aggregates, timeout=15)
        if r.status_code == 200:
            return r.json()
        else:
            try:
                return {"error": f"{r.status_code} {r.text}"}
            except Exception:
                return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# --------------------
# Heuristics
# --------------------
def compute_heuristic_scores(weather: Dict[str,Any]) -> Dict[str,float]:
    scores = {"very_hot":0.0,"very_cold":0.0,"very_windy":0.0,"very_wet":0.0,"very_uncomfortable":0.0}
    if not weather:
        return scores
    hourly = weather.get("hourly", {})
    temps = np.array(hourly.get("temperature_2m", []) or [0.0], dtype=float)
    app_temps = np.array(hourly.get("apparent_temperature", []) or temps, dtype=float)
    humid = np.array(hourly.get("relative_humidity_2m", []) or [0.0], dtype=float)
    winds = np.array(hourly.get("wind_speed_10m", []) or [0.0], dtype=float)
    precip_prob = np.array(hourly.get("precipitation_probability", []) or [0.0], dtype=float)

    max_temp = float(np.max(temps)) if temps.size else 0.0
    min_temp = float(np.min(temps)) if temps.size else 0.0
    avg_precip = float(np.mean(precip_prob)) if precip_prob.size else 0.0
    max_wind = float(np.max(winds)) if winds.size else 0.0
    mean_hum = float(np.mean(humid)) if humid.size else 0.0
    max_app = float(np.max(app_temps)) if app_temps.size else max_temp

    scores["very_hot"] = min(1.0, max(0.0, (max_temp - 30.0) / 15.0))
    scores["very_cold"] = min(1.0, max(0.0, (0.0 - min_temp) / 30.0))
    scores["very_windy"] = min(1.0, max(0.0, (max_wind - 10.0) / 20.0))
    scores["very_wet"] = min(1.0, max(0.0, avg_precip / 100.0))
    discomfort = 0.0
    if max_app > 25:
        discomfort += (max_app - 25.0) / 20.0
    if mean_hum > 50:
        discomfort += (mean_hum - 50.0) / 100.0
    scores["very_uncomfortable"] = min(1.0, max(0.0, discomfort))
    return scores

def pct(v: float) -> int:
    return int(round(100 * float(v)))

# --------------------
# Small helpers: format prediction
# --------------------
def sigmoid(x: float) -> float:
    try:
        return 1 / (1 + math.exp(-float(x)))
    except Exception:
        return 0.5

def interpret_prediction(pred_raw: Any, probs: Optional[Any]=None) -> Dict[str,Any]:
    """
    Normalize model output for user:
    - if model returns numeric (regressor), map via sigmoid -> percent
    - if model returns class/probs, use probs if present
    Returns dict with keys: label, percent, details
    """
    result = {"label":"unknown", "percent":None, "details":{}}
    try:
        # if probs present and looks like list of probabilities -> show top class
        if probs:
            # assume probs is list or list-of-lists with first element
            if isinstance(probs, list) and len(probs) and isinstance(probs[0], (list,tuple)):
                p = np.array(probs[0], dtype=float)
                top_idx = int(np.argmax(p))
                result["label"] = f"class_{top_idx}"
                result["percent"] = float(p[top_idx])
                result["details"]["probs"] = p.tolist()
                return result
            elif isinstance(probs, list):
                p = np.array(probs, dtype=float)
                # if single-d array
                if p.size == 1:
                    # probability for positive class
                    result["percent"] = float(p[0])
                    result["label"] = "positive_prob"
                    return result
        # Else, numeric prediction -> map with sigmoid to percent for user friendliness
        val = float(pred_raw)
        prob = sigmoid(val)  # 0..1
        # interpret percent thresholds
        if prob >= 0.75:
            label = "High"
        elif prob >= 0.55:
            label = "Likely"
        elif prob >= 0.45:
            label = "Possible"
        elif prob >= 0.25:
            label = "Unlikely"
        else:
            label = "Very unlikely"
        result["label"] = label
        result["percent"] = prob
        result["details"]["raw_value"] = val
        return result
    except Exception:
        return result

# --------------------
# UI layout
# --------------------
st.markdown("<style>.full-map > div { height: 90vh !important; }</style>", unsafe_allow_html=True)
left_col, right_col = st.columns([6, 1.2])

with left_col:
    st.title("NASA Weather Globe — Hybrid")
    search_input = st.text_input(I18N[lang]["search_placeholder"], "")
    suggestions = []
    q = search_input.strip()
    if q:
        def coords_try(s: str) -> Optional[Tuple[float, float]]:
            s2 = s.replace(",", " ").strip()
            parts = s2.split()
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1])
                except Exception:
                    return None
            return None
        c = coords_try(q)
        if c:
            lat_q, lon_q = c
            st.session_state.pending = {"lat": float(lat_q), "lon": float(lon_q), "city": f"{lat_q:.4f},{lon_q:.4f}"}
            st.session_state.nearby = []
            st.session_state.weather = get_weather_open_meteo(lat_q, lon_q, days=7)
            st.session_state.pending["altitude_m"] = get_altitude_opentopo(lat_q, lon_q)
        else:
            mask = cities_df["display_name"].str.lower().str.contains(q.lower())
            local_hits = cities_df[mask].head(200)
            for _, r in local_hits.iterrows():
                suggestions.append(("LOCAL", r["display_name"], float(r["lat"]), float(r["lon"])))
            if len(suggestions) < 20:
                try:
                    nom = nominatim_search(q, limit=10)
                    for it in nom:
                        dd = it.get("display_name","")
                        latn = float(it.get("lat"))
                        lonn = float(it.get("lon"))
                        suggestions.append(("NOM", dd, latn, lonn))
                except Exception:
                    pass

    if suggestions:
        opts = [f"{kind}|{name}|{lat}|{lon}" for (kind,name,lat,lon) in suggestions]
        sel = st.selectbox("Suggestions (choose):", ["---"] + opts, key="live_suggest")
        if sel and sel != "---":
            _, name, lat_s, lon_s = sel.split("|",3)
            st.session_state.pending = {"lat": float(lat_s), "lon": float(lon_s), "city": name}
            st.session_state.nearby = []
            st.session_state.weather = get_weather_open_meteo(float(lat_s), float(lon_s), days=7)
            st.session_state.pending["altitude_m"] = get_altitude_opentopo(float(lat_s), float(lon_s))
    else:
        if q:
            st.write("No local suggestions. Try another query or use 'Search (Nominatim fallback)'.")

    if st.button("Search (Nominatim fallback)"):
        if q:
            nom = nominatim_search(q, limit=20)
            if nom:
                opts = [f'{item.get("display_name","")}|{item.get("lat")}|{item.get("lon")}' for item in nom]
                sel = st.selectbox("Nominatim matches:", ["---"] + opts, key="nom_matches")
                if sel and sel != "---":
                    disp, lat_s, lon_s = sel.split("|",2)
                    st.session_state.pending = {"lat": float(lat_s), "lon": float(lon_s), "city": disp}
                    st.session_state.nearby = []
                    st.session_state.weather = get_weather_open_meteo(float(lat_s), float(lon_s), days=7)
                    st.session_state.pending["altitude_m"] = get_altitude_opentopo(float(lat_s), float(lon_s))

with right_col:
    st.markdown(f"### {I18N[lang]['map_options']}")
    tile_date = st.date_input(I18N[lang]["modis_date"], datetime.date.today())
    show_modis = st.checkbox(I18N[lang]["show_modis"], value=True)
    base_map = st.selectbox(I18N[lang]["base_map"], ["Hybrid (Esri satellite)", "Streets (OSM)"])
    st.markdown(f"City DB: {len(cities_df):,} entries")

# Sidebar legend
st.sidebar.header(I18N[lang]["legend_caption"])
legend_svg = make_legend_svg(TEMP_MIN, TEMP_MAX, width=320, height=56)
legend_b64 = base64.b64encode(legend_svg.encode("utf-8")).decode("utf-8")
st.sidebar.image(f"data:image/svg+xml;base64,{legend_b64}", caption=f"{TEMP_MIN}°C … {TEMP_MAX}°C")
highlight_temp = st.sidebar.slider("Highlight near (°C)", min_value=TEMP_MIN, max_value=TEMP_MAX, value=0, step=1)
highlight_delta = st.sidebar.slider("± range (°C)", min_value=0, max_value=20, value=2, step=1)
hmin = highlight_temp - highlight_delta; hmax = highlight_temp + highlight_delta

# Active = pending if exists else confirmed
active = st.session_state.pending if st.session_state.pending else st.session_state.confirmed

# --------------------
# Map render & interactions
# --------------------
st.subheader("🗺 Map — click to pick coordinates, drag marker to move, then Confirm")

if USE_FOLIUM and USE_STREAMLIT_FOLIUM:
    m = folium.Map(location=[active["lat"], active["lon"]], zoom_start=5, tiles=None, control_scale=True)
    if base_map.startswith("Hybrid"):
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri World Imagery",
            control=False,
            max_zoom=19,
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="© OpenStreetMap contributors",
            name="OpenStreetMap",
            control=False,
            max_zoom=19,
        ).add_to(m)

    if show_modis:
        tpl = NASA_GIBS_TEMPLATE.format(layer=GIBS_LAYER, date=tile_date.isoformat())
        folium.raster_layers.TileLayer(
            tiles=tpl,
            attr="NASA GIBS MODIS",
            name="MODIS LST",
            overlay=True,
            control=False,
            opacity=0.6,
            max_zoom=9
        ).add_to(m)

    if st.session_state.nearby:
        cluster = MarkerCluster().add_to(m)
        for it in st.session_state.nearby:
            col = temp_to_hex(it.get("temp"))
            popup_html = f"{it['display_name']}<br>{it.get('dist_km',0):.1f} km"
            folium.CircleMarker(
                location=(it["lat"], it["lon"]),
                radius=6, color=col, fill=True, fill_color=col, fill_opacity=0.9, popup=popup_html
            ).add_to(cluster)

    popup_brief = None
    if st.session_state.weather and st.session_state.confirmed:
        try:
            w = st.session_state.weather
            cur = w.get("current_weather", {})
            temp_now = cur.get("temperature")
            wind = cur.get("windspeed")
            popup_brief = f"<b>{st.session_state.confirmed.get('city','')}</b><br>🌡 {temp_now}°C — 💨 {wind} m/s"
        except Exception:
            popup_brief = None

    marker = folium.Marker(
        location=[active["lat"], active["lon"]],
        draggable=True,
        tooltip="Drag marker -> map picks coordinates",
        icon=folium.Icon(color="red", icon="map-marker")
    )
    if popup_brief:
        marker.add_child(folium.Popup(popup_brief, max_width=320))
    marker.add_to(m)

    legend_html = f'''
        <div style="position: fixed; bottom: 12px; right: 12px; z-index:9999; background: rgba(255,255,255,0.95); padding:6px; border-radius:6px; box-shadow:0 0 8px rgba(0,0,0,0.15);">
          <img src="data:image/svg+xml;base64,{legend_b64}" style="width:320px; height:56px; display:block;">
        </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    map_data = st_folium(m, width="100%", height=MAP_HEIGHT, returned_objects=["last_clicked", "last_marker", "last_marker_drag"])
    if map_data:
        clicked = map_data.get("last_clicked")
        if clicked:
            latc = clicked.get("lat"); lonc = clicked.get("lng")
            if latc is not None and lonc is not None:
                st.session_state.pending = {"lat": float(latc), "lon": float(lonc), "city": f"{latc:.5f},{lonc:.5f}"}
                d = haversine_array(latc, lonc, city_lats, city_lons)
                idxs = np.argsort(d)[:NEAREST_SUGGESTIONS]
                nearest = []
                for i in idxs:
                    nearest.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
                st.session_state.nearby = nearest
                st.session_state.weather = get_weather_open_meteo(latc, lonc, days=7)
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(latc, lonc)

        drag = map_data.get("last_marker_drag") or map_data.get("last_marker")
        if drag:
            latm = drag.get("lat"); lonm = drag.get("lng")
            if latm is not None and lonm is not None:
                st.session_state.pending = {"lat": float(latm), "lon": float(lonm), "city": f"{latm:.5f},{lonm:.5f}"}
                d = haversine_array(latm, lonm, city_lats, city_lons)
                idxs = np.argsort(d)[:NEAREST_SUGGESTIONS]
                nearest = []
                for i in idxs:
                    nearest.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
                st.session_state.nearby = nearest
                st.session_state.weather = get_weather_open_meteo(latm, lonm, days=7)
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(latm, lonm)

else:
    st.warning("Folium or streamlit_folium not available — using pydeck fallback (less interactive). Install folium & streamlit_folium for draggable markers.")
    if USE_PYDECK:
        esri = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        layers = [pdk.Layer("TileLayer", data=esri, tileSize=256)]
        pt = pd.DataFrame([{"lat": active["lat"], "lon": active["lon"]}])
        layers.append(pdk.Layer("ScatterplotLayer", data=pt, get_position='[lon, lat]', get_color='[255,0,0,255]', get_radius=100000))
        view = pdk.ViewState(latitude=active["lat"], longitude=active["lon"], zoom=4, pitch=30)
        deck = pdk.Deck(layers=layers, initial_view_state=view, map_style=None)
        st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    else:
        st.error("No mapping library available. Install folium or pydeck.")

# --------------------
# Sidebar: details & user-friendly ML display
# --------------------
if st.session_state.pending or st.session_state.confirmed:
    with st.sidebar:
        st.header("📍 Point & Weather")
        cur = st.session_state.pending if st.session_state.pending else st.session_state.confirmed
        label = I18N[lang]["pending"] if st.session_state.pending else I18N[lang]["confirmed"]
        st.write(f"**{label}**: {cur.get('city','Unknown')}")
        st.write(f"{I18N[lang]['coords_label']}: {cur['lat']:.6f}, {cur['lon']:.6f}")

        # nearest choices
        if st.session_state.nearby:
            st.markdown(f"### {I18N[lang]['nearest_label']}")
            for i, item in enumerate(st.session_state.nearby):
                btn_label = f"{item['display_name']} — {item['dist_km']:.1f} km"
                if st.button(btn_label, key=f"near_{i}"):
                    st.session_state.pending = {"lat": item["lat"], "lon": item["lon"], "city": item["display_name"]}
                    st.session_state.weather = get_weather_open_meteo(item["lat"], item["lon"], days=7)
                    st.session_state.pending["altitude_m"] = get_altitude_opentopo(item["lat"], item["lon"])
                    st.session_state.nearby = []

        # action buttons
        c1, c2, c3 = st.columns([1,1,1])
        if c1.button(I18N[lang]["confirm"]):
            if st.session_state.pending:
                st.session_state.confirmed = st.session_state.pending.copy()
                st.session_state.pending = None
                label_city = st.session_state.confirmed.get("city")
                if label_city and label_city not in st.session_state.mru:
                    st.session_state.mru.append(label_city)
        if c2.button(I18N[lang]["cancel"]):
            st.session_state.pending = None
            st.session_state.nearby = []
            st.session_state.weather = None
        if c3.button(I18N[lang]["nearby_quick"]):
            p = st.session_state.pending if st.session_state.pending else st.session_state.confirmed
            lat0 = p["lat"]; lon0 = p["lon"]
            d = haversine_array(lat0, lon0, city_lats, city_lons)
            idxs = np.argsort(d)[:200] if len(d)>0 else []
            nearby = []
            for i in idxs:
                nearby.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
            st.session_state.nearby = nearby

        st.markdown("---")
        if st.session_state.weather:
            w = st.session_state.weather
            cur = w.get("current_weather", {})
            if cur:
                st.metric("🌡 Temperature (now)", f"{cur.get('temperature')} °C")
                st.metric("💨 Wind (m/s)", f"{cur.get('windspeed')} m/s")

            hourly = w.get("hourly", {})
            if hourly and "time" in hourly:
                df_hour = pd.DataFrame({
                    "time": pd.to_datetime(hourly.get("time", [])),
                    "temperature": hourly.get("temperature_2m", []),
                    "apparent": hourly.get("apparent_temperature", []),
                    "humidity": hourly.get("relative_humidity_2m", []),
                    "precip_prob": hourly.get("precipitation_probability", []),
                    "uv": hourly.get("uv_index", []),
                    "cloud": hourly.get("cloudcover", []),
                    "wind": hourly.get("wind_speed_10m", []),
                })
                st.markdown(f"### {I18N[lang]['next_48h']}")
                st.line_chart(df_hour.set_index("time")[["temperature","precip_prob"]].head(48))
                st.dataframe(df_hour.head(12).set_index("time"))
                csv = df_hour.to_csv(index=False).encode("utf-8")
                st.download_button(I18N[lang]["download_csv"], data=csv, file_name="hourly.csv", mime="text/csv")

            daily = w.get("daily", {})
            if daily and "time" in daily:
                df_daily = pd.DataFrame({
                    "date": pd.to_datetime(daily.get("time", [])),
                    "t_min": daily.get("temperature_2m_min", []),
                    "t_max": daily.get("temperature_2m_max", []),
                    "sunrise": daily.get("sunrise", []),
                    "sunset": daily.get("sunset", []),
                })
                st.markdown(f"### {I18N[lang]['seven_day']}")
                st.dataframe(df_daily.set_index("date").head(7))

            alt_local = None
            if st.session_state.confirmed and "altitude_m" in st.session_state.confirmed:
                alt_local = st.session_state.confirmed.get("altitude_m")
            if st.session_state.pending and "altitude_m" in st.session_state.pending:
                alt_local = st.session_state.pending.get("altitude_m")
            if alt_local is not None:
                st.write(f"⛰ Altitude (OpenTopoData): {alt_local:.1f} m")

            # --------------------
            # ML & heuristics: compute + display user-friendly summary
            # --------------------
            aggregates = extract_simple_features_from_weather(st.session_state.weather)
            pred_resp = None
            if aggregates:
                pred_resp = call_backend_prediction(aggregates)

            heur = compute_heuristic_scores(st.session_state.weather)

            st.markdown("---")
            st.subheader("🔮 Weather summary")

            # User-friendly summary card
            summary_col1, summary_col2 = st.columns([2,1])
            with summary_col1:
                # Short human message built from heuristics and ML
                # Determine main "headline"
                headline = ""
                subtxt = ""
                if pred_resp and isinstance(pred_resp, dict) and pred_resp.get("error"):
                    headline = "ML unavailable"
                    subtxt = "Backend returned an error. Showing heuristic summary."
                else:
                    # interpret ML if present
                    if pred_resp and isinstance(pred_resp, dict) and pred_resp.get("prediction") is not None:
                        interp = interpret_prediction(pred_resp.get("prediction"), pred_resp.get("probs"))
                        pct_val = interp.get("percent")
                        if pct_val is not None:
                            pct_show = int(round(100 * float(pct_val)))
                            headline = f"Chance of adverse conditions: {pct_show}%"
                            subtxt = f"ML says: {interp.get('label')}"
                        else:
                            headline = f"Model output: {str(pred_resp.get('prediction'))}"
                            subtxt = "Interpreting model output..."
                    else:
                        # fallback to heuristics: take max of heuristics as main
                        mx_key = max(heur.keys(), key=lambda k: heur.get(k,0))
                        mx_val = heur.get(mx_key, 0.0)
                        headline = f"Heuristic risk: {pct(mx_val)}%"
                        subtxt = f"Top factor: {mx_key.replace('_',' ')}"

                st.markdown(f"### {headline}")
                st.write(subtxt)

                # Small bullets with main numeric metrics
                try:
                    cur_weather = st.session_state.weather.get("current_weather", {})
                    st.write(f"**Now:** {cur_weather.get('temperature','?')} °C  — wind {cur_weather.get('windspeed','?')} m/s")
                except Exception:
                    pass

                # Short human advice based on heuristics + ml
                advice = []
                if heur.get("very_windy",0) >= 0.7:
                    advice.append("⚠️ Strong winds — secure light objects.")
                if heur.get("very_wet",0) >= 0.5:
                    advice.append("🌧️ High precip probability — bring umbrella.")
                if heur.get("very_uncomfortable",0) >= 0.6:
                    advice.append("🥵 Conditions uncomfortable — stay hydrated.")
                if pred_resp and isinstance(pred_resp, dict) and pred_resp.get("prediction") is not None:
                    interp = interpret_prediction(pred_resp.get("prediction"), pred_resp.get("probs"))
                    # if ML predicts high risk
                    if interp.get("percent") and interp.get("percent") >= 0.75:
                        advice.append("🔴 ML: High risk — consider changing plans.")
                if not advice:
                    advice.append("✅ No special precautions needed now.")
                for a in advice:
                    st.write(a)

            with summary_col2:
                st.write("Quick status")
                st.metric("Heur. wind", f"{pct(heur.get('very_windy',0))}%")
                st.metric("Heur. wet", f"{pct(heur.get('very_wet',0))}%")
                st.metric("Heur. discomfort", f"{pct(heur.get('very_uncomfortable',0))}%")

            # Detailed debug section (hidden by default)
            with st.expander("Показати деталі / debug (для розробника)"):
                st.write("Aggregates sent to backend:")
                st.json(aggregates)
                st.write("---")
                st.write("Heuristic scores:")
                st.json(heur)

                st.write("---")
                st.write("Backend response:")
                if pred_resp is None:
                    st.write("No backend response (backend unreachable or no aggregates).")
                else:
                    st.json(pred_resp)
                    # if note present (feature adaptation), explain in human terms
                    note = pred_resp.get("note") if isinstance(pred_resp, dict) else None
                    if note:
                        if "trimmed" in note:
                            st.warning("Model expected fewer features; some inputs were ignored. Ask ML team for feature order.")
                        if "padded" in note:
                            st.info("Model expected more features; missing features were padded with zeros.")
                    # original / used features
                    if isinstance(pred_resp, dict) and pred_resp.get("original_features") is not None:
                        st.write("Original features sent:")
                        st.write(pred_resp.get("original_features"))
                    if isinstance(pred_resp, dict) and pred_resp.get("used_features") is not None:
                        st.write("Features used by model:")
                        st.write(pred_resp.get("used_features"))

        else:
            st.info(I18N[lang]["no_weather"])

        st.markdown("---")
        st.subheader(I18N[lang]["recent"])
        for r in st.session_state.mru[-8:][::-1]:
            if st.button(f"→ {r}", key=f"mru_{r}"):
                mask = cities_df["display_name"] == r
                if mask.any():
                    rr = cities_df[mask].iloc[0]
                    st.session_state.pending = {"lat": float(rr["lat"]), "lon": float(rr["lon"]), "city": r}
                    st.session_state.weather = get_weather_open_meteo(rr["lat"], rr["lon"], days=7)
                    st.session_state.pending["altitude_m"] = get_altitude_opentopo(rr["lat"], rr["lon"])

# language toggle small
def lang_toggle_ui():
    if st.button(I18N[lang]["lang_toggle"]):
        pass  # placeholder

lang_toggle_ui()

st.markdown("---")
st.caption("Local cities only (data/cities.json). For best draggable marker experience install folium & streamlit-folium.")
