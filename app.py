# app.py
""" NASA Weather Globe — final merged app (Streamlit frontend)
Features:
 - Local data/cities.json (fallback list included)
 - Folium + streamlit_folium for draggable marker & clicks (pydeck fallback)
 - Search by local DB or coordinates (Nominatim fallback)
 - Open-Meteo for weather, OpenTopoData for altitude (best-effort)
 - Heuristic risk scores + call to backend /predict (BACKEND_URL env)
 - Friendly UI, CSV export, MRU, nearest suggestions
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import os
import textwrap
import base64
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests

# Optional mapping libs
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

# -------------------------
# Config & constants
# -------------------------
APP_TITLE = "NASA Weather Globe"
APP_ICON = "🌍"
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="auto")

MAP_HEIGHT = 760
LOCAL_DATA_DIR = Path("data")
LOCAL_CITIES_FILE = LOCAL_DATA_DIR / "cities.json"

GIBS_LAYER = "MODIS_Terra_Land_Surface_Temp_Day"
NASA_GIBS_TEMPLATE = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{layer}/default/{date}/"
    "GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.png"
)

TEMP_MIN_DEFAULT = -50
TEMP_MAX_DEFAULT = 50
NEAREST_SUGGESTIONS = 6

USER_AGENT = "streamlit-nasa-globe/1.0 (+https://example)"

# Backend URL (set via env in Render). Default to localhost for local dev.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

# -------------------------
# i18n
# -------------------------
LANGS = ["en", "ua"]
DEFAULT_LANG = "ua"
I18N: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "NASA Weather Globe",
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
        "lang_toggle": "EN / УКР",
        "loading_local": "Loading local city DB...",
        "local_missing": "Local cities.json not found — using small fallback.",
        "search_button": "Search",
        "search_nominatim": "Search (Nominatim fallback)",
    },
    "ua": {
        "title": "NASA Weather Globe",
        "search_placeholder": "Пошук: місто або координати (напр. Київ або 50.45,30.52)",
        "map_options": "Налаштування карти",
        "modis_date": "Дата MODIS",
        "show_modis": "Показати MODIS LST (NASA GIBS)",
        "base_map": "Базова карта",
        "pending": "ЧЕК",
        "confirmed": "ПІДТВЕРДЖЕНО",
        "confirm": "Підтвердити",
        "cancel": "Скасувати",
        "nearby_quick": "Поблизу (швидко)",
        "download_csv": "Завантажити hourly CSV",
        "next_48h": "Наступні 48 год",
        "seven_day": "7 днів",
        "no_weather": "Підтвердіть точку, щоб завантажити погоду.",
        "recent": "Останні вибори",
        "nearest_label": "Найближчі міста (оберіть)",
        "coords_label": "Координати",
        "lang_toggle": "EN / УКР",
        "loading_local": "Завантаження локальної БД міст...",
        "local_missing": "Файл data/cities.json не знайдено — використовується малий список.",
        "search_button": "Пошук",
        "search_nominatim": "Пошук (Nominatim fallback)",
    },
}

# -------------------------
# Small fallback cities
# -------------------------
SMALL_FALLBACK = [
    {"display_name": "Kyiv, Ukraine", "lat": 50.4501, "lon": 30.5234},
    {"display_name": "Moscow, Russia", "lat": 55.7558, "lon": 37.6173},
    {"display_name": "Warsaw, Poland", "lat": 52.2297, "lon": 21.0122},
    {"display_name": "Berlin, Germany", "lat": 52.52, "lon": 13.405},
    {"display_name": "Paris, France", "lat": 48.8566, "lon": 2.3522},
    {"display_name": "Rome, Italy", "lat": 41.9028, "lon": 12.4964},
    {"display_name": "Istanbul, Turkey", "lat": 41.0082, "lon": 28.9784},
    {"display_name": "Budapest, Hungary", "lat": 47.4979, "lon": 19.0402},
    {"display_name": "Bucharest, Romania", "lat": 44.4268, "lon": 26.1025},
    {"display_name": "Prague, Czechia", "lat": 50.0755, "lon": 14.4378},
]

# -------------------------
# Helpers
# -------------------------
def temp_to_hex(t: Optional[float], vmin: int = TEMP_MIN_DEFAULT, vmax: int = TEMP_MAX_DEFAULT) -> str:
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

def haversine_array(lat1: float, lon1: float, lat2_array: np.ndarray, lon2_array: np.ndarray) -> np.ndarray:
    R = 6371.0
    lat1r = np.radians(lat1); lon1r = np.radians(lon1)
    lat2r = np.radians(np.array(lat2_array, dtype=float))
    lon2r = np.radians(np.array(lon2_array, dtype=float))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------------
# Network helpers (cached)
# -------------------------
@st.cache_data(ttl=60*60)
def nominatim_search(q: str, limit: int = 12) -> list:
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": limit},
            headers={"User-Agent": USER_AGENT},
            timeout=8,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        return []
    return []

@st.cache_data(ttl=60*60*3)
def get_weather_open_meteo(lat: float, lon: float, days: int = 7) -> Optional[dict]:
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,apparent_temperature,relative_humidity_2m,precipitation_probability,wind_speed_10m,cloudcover,uv_index"
            f"&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset"
            f"&current_weather=true&forecast_days={days}&timezone=auto"
        )
        r = requests.get(url, timeout=12, headers={"User-Agent": USER_AGENT})
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

@st.cache_data(ttl=60*60*24)
def get_altitude_opentopo(lat: float, lon: float) -> Optional[float]:
    try:
        r = requests.get(
            "https://api.opentopodata.org/v1/test-dataset",
            params={"locations": f"{lat},{lon}"},
            timeout=8,
            headers={"User-Agent": USER_AGENT},
        )
        if r.status_code == 200:
            j = r.json()
            if "results" in j and j["results"]:
                return j["results"][0].get("elevation")
    except Exception:
        return None
    return None

# -------------------------
# Local cities loading
# -------------------------
def load_local_cities() -> pd.DataFrame:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_CITIES_FILE.exists():
        try:
            with open(LOCAL_CITIES_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            df = pd.DataFrame(raw)
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
            df["lat"] = pd.to_numeric(df.get("lat", None), errors="coerce")
            df["lon"] = pd.to_numeric(df.get("lon", None), errors="coerce")
            df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
            df = df[["display_name", "lat", "lon"]]
            return df
        except Exception:
            try:
                st.warning(I18N[get_lang()]["local_missing"])
            except Exception:
                pass
            return pd.DataFrame(SMALL_FALLBACK)
    else:
        try:
            st.info(I18N[get_lang()]["local_missing"])
        except Exception:
            pass
        return pd.DataFrame(SMALL_FALLBACK)

# -------------------------
# Session state init
# -------------------------
if "lang" not in st.session_state:
    st.session_state.lang = DEFAULT_LANG
if "confirmed" not in st.session_state:
    st.session_state.confirmed = {"lat": 50.4501, "lon": 30.5234, "city": "Kyiv, Ukraine"}
if "pending" not in st.session_state:
    st.session_state.pending = None
if "weather" not in st.session_state:
    st.session_state.weather = None
if "nearby" not in st.session_state:
    st.session_state.nearby = []
if "mru" not in st.session_state:
    st.session_state.mru = []
if "map_used" not in st.session_state:
    st.session_state.map_used = False
if "forecast_days" not in st.session_state:
    st.session_state["forecast_days"] = 7

def get_lang() -> str:
    return st.session_state.lang if "lang" in st.session_state else DEFAULT_LANG

lang = get_lang()

# Load cities
with st.spinner(I18N[lang]["loading_local"]):
    cities_df = load_local_cities()

city_lats = cities_df["lat"].to_numpy() if not cities_df.empty else np.array([])
city_lons = cities_df["lon"].to_numpy() if not cities_df.empty else np.array([])
city_disp = cities_df["display_name"].to_numpy() if not cities_df.empty else np.array([])

# -------------------------
# Helper: extract features from weather (same as backend expects)
# -------------------------
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

# -------------------------
# ML backend call
# -------------------------
def call_backend_prediction(aggregates: Dict[str,Any], timeout: int = 12) -> Optional[Dict[str,Any]]:
    """Call backend /predict and return parsed JSON or dict with error."""
    if not aggregates:
        return None
    try:
        url = f"{BACKEND_URL}/predict"
        r = requests.post(url, json=aggregates, timeout=timeout, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        return {"error": f"Request error: {e}"}
    try:
        if r.status_code == 200:
            return r.json()
        else:
            # Attempt parse JSON error response
            try:
                return {"error": f"{r.status_code} {r.text}"}
            except Exception:
                return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": f"Response parsing failed: {e}"}

# -------------------------
# Heuristics: compute simple scores 0..1
# -------------------------
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

# -------------------------
# UI CSS + header layout
# -------------------------
st.markdown(
    f"""
    <style>
    .title-area {{ display:flex; align-items:center; gap:18px; padding:18px 28px; }}
    .app-title {{ font-size:28px; font-weight:700; margin:0; line-height:1; }}
    .search-row {{ display:block; padding: 6px 28px 18px 28px; }}
    .search-box {{ width:100%; }}
    .search-subrow {{ display:flex; gap:12px; align-items:center; margin-top:8px; }}
    .search-subrow .stButton button {{ white-space:nowrap; min-width:96px; }}
    .map-area {{ padding: 6px 28px 40px 28px; }}
    .sidebar .element-container {{ padding-top: 8px !important; }}
    .stFMap {{ height: {MAP_HEIGHT}px !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
col_a, col_b, col_c = st.columns([2, 7, 1])
with col_a:
    # --- logo: safe base64 data-URI (fixed) ---
    svg = textwrap.dedent("""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="50" fill="#0B3D91"/>
      <path fill="#FC3D21" d="M50 10C28 10 10 28 10 50s18 40 40 40 40-18 40-40S72 10 50 10zm0 74c-18.7 0-34-15.3-34-34S31.3 16 50 16s34 15.3 34 34-15.3 34-34 34z"/>
     <path fill="#fff" d="M73 61.5c-1.2 0-2.3-.2-3.4-.6-1.1-.4-2.1-1.1-3-2.1-.9-1-1.6-2.3-2-3.8l-2.3-7.5h4.2l2 6.4c.4 1.3.9 2.2 1.5 2.8.6.6 1.3.9 2.2.9 1 0 1.7-.3 2.3-.9.6-.6.9-1.5.9-2.7 0-.9-.2-1.7-.7-2.5-.5-.8-1.1-1.4-1.8-1.9-.8-.5-1.8-.9-3.1-1.2l-1.3-.3c-1.8-.4-3.2-1.1-4.2-2.1-1-.9-1.7-2.1-2.2-3.4-.4-1.3-.7-2.8-.7-4.3 0-2.2.4-4.2 1.3-6 1-1.7 2.3-3 3.9-3.9 1.7-.9 3.6-1.4 5.7-1.4 1.6 0 3.1.2 4.4.7 1.3.5 2.5 1.2 3.5 2.2 1 .9 1.7 2.1 2.2 3.4.5 1.3.7 2.8.7 4.3H73c0-1.3-.3-2.4-.8-3.3-.5-.9-1.2-1.6-2-2.1-.8-.5-1.8-.7-2.9-.7-1.6 0-2.9.5-3.9 1.6-1 1.1-1.5 2.5-1.5 4.2 0 1.3.2 2.3.7 3.2.5.8 1.2 1.5 2.1 2 .9.5 2 .8 3.3 1.1l1.4.3c2.5.5 4.4 1.6 5.9 3.1 1.5 1.5 2.2 3.5 2.2 6.1 0 2.1-.4 3.9-1.3 5.6-.9 1.6-2.2 2.9-3.8 3.8-1.7.8-3.6 1.3-5.8 1.3z"/>
    </svg>
    """).strip()
    svg_b64 = base64.b64encode(svg.encode('utf-8')).decode('ascii')
    data_uri = f"data:image/svg+xml;base64,{svg_b64}"
    st.image(data_uri, width=56)
    st.markdown(f"<div class='app-title'>{I18N[lang]['title']}</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:14px; color:var(--secondary-text-color)'>{I18N[lang]['search_placeholder']}</div>", unsafe_allow_html=True)

    # --- SEARCH: input full width, button + lang under it ---
    st.markdown("<div class='search-row'>", unsafe_allow_html=True)
    search_input = st.text_input("", key="header_search", placeholder=I18N[lang]["search_placeholder"], label_visibility="collapsed")
    # sub-row with button + lang selector (prevents button wrapping)
    sub_col_left, sub_col_right = st.columns([1, 0.28])
    with sub_col_left:
        if st.button(I18N[lang]["search_button"], key="header_search_btn"):
            qval = st.session_state.get("header_search", "")
            # reuse same processing function defined below (call directly)
            # We'll implement processing inline to avoid duplication
            q = (qval or "").strip()
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
                    st.session_state.pending = {"lat": float(lat_q), "lon": float(lon_q), "city": f"{lat_q:.5f},{lon_q:.5f}"}
                    st.session_state.nearby = []
                    st.session_state.weather = get_weather_open_meteo(lat_q, lon_q, days=st.session_state.get("forecast_days", 7))
                    st.session_state.pending["altitude_m"] = get_altitude_opentopo(lat_q, lon_q)
                else:
                    # local substring search
                    suggestions_local = []
                    if not cities_df.empty:
                        mask = cities_df["display_name"].str.lower().str.contains(q.lower())
                        local_hits = cities_df[mask].head(200)
                        for _, r in local_hits.iterrows():
                            suggestions_local.append(("LOCAL", r["display_name"], float(r["lat"]), float(r["lon"])))
                    suggestions = suggestions_local
                    if len(suggestions_local) < 10:
                        nom = nominatim_search(q, limit=8)
                        for it in nom:
                            try:
                                name = it.get("display_name", "")
                                latn = float(it.get("lat"))
                                lonn = float(it.get("lon"))
                                suggestions.append(("NOM", name, latn, lonn))
                            except Exception:
                                continue
                    if suggestions:
                        kind, name, latn, lonn = suggestions[0]
                        st.session_state.pending = {"lat": float(latn), "lon": float(lonn), "city": name}
                        st.session_state.nearby = []
                        st.session_state.weather = get_weather_open_meteo(latn, lonn, days=st.session_state.get("forecast_days", 7))
                        st.session_state.pending["altitude_m"] = get_altitude_opentopo(latn, lonn)
                    else:
                        st.warning("No matches found for query.")
    with sub_col_right:
        lang_select = st.selectbox("", options=["ua", "en"], index=0 if lang == "ua" else 1, key="lang_select_small")
        if lang_select != st.session_state.lang:
            st.session_state.lang = lang_select
            lang = get_lang()
    st.markdown("</div>", unsafe_allow_html=True)

with col_c:
    st.write("")

st.markdown("<hr style='margin:8px 0 12px 0'>", unsafe_allow_html=True)

st.markdown("<div style='padding-left:28px; padding-top:8px'>", unsafe_allow_html=True)
if st.button(I18N[lang]["search_nominatim"], key="nominatim_fallback"):
    qval = st.session_state.get("header_search", "").strip()
    if qval:
        nom = nominatim_search(qval, limit=20)
        if nom:
            opts = [f'{item.get("display_name","")}|{item.get("lat")}|{item.get("lon")}' for item in nom]
            sel = st.selectbox("Nominatim matches:", ["---"] + opts, key="nominatim_matches")
            if sel and sel != "---":
                disp, lat_s, lon_s = sel.split("|", 2)
                st.session_state.pending = {"lat": float(lat_s), "lon": float(lon_s), "city": disp}
                st.session_state.nearby = []
                st.session_state.weather = get_weather_open_meteo(float(lat_s), float(lon_s), days=st.session_state.get("forecast_days", 7))
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(float(lat_s), float(lon_s))
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar + controls
# -------------------------
with st.sidebar:
    st.header(I18N[lang]["map_options"])
    tile_date = st.date_input(I18N[lang]["modis_date"], datetime.date.today())
    show_modis = st.checkbox(I18N[lang]["show_modis"], value=True)
    base_map = st.selectbox(I18N[lang]["base_map"], ["Hybrid (Esri satellite)", "Streets (OSM)"])
    st.markdown(f"City DB: {len(cities_df):,} entries")
    st.markdown("---")
    forecast_days = st.slider("Forecast days", min_value=1, max_value=14, value=st.session_state["forecast_days"], key="sidebar_forecast_days")
    st.session_state["forecast_days"] = forecast_days
    st.subheader("Selected")
    if st.session_state.confirmed:
        st.write("Confirmed:", st.session_state.confirmed.get("city", "Unknown"))
        st.write(f"{I18N[lang]['coords_label']}: {st.session_state.confirmed['lat']:.6f}, {st.session_state.confirmed['lon']:.6f}")
    if st.session_state.pending:
        st.write("Pending:", st.session_state.pending.get("city", "Unknown"))
        st.write(f"{I18N[lang]['coords_label']}: {st.session_state.pending['lat']:.6f}, {st.session_state.pending['lon']:.6f}")
    st.markdown("---")
    if st.button("Center to confirmed", key="center_confirmed_btn"):
        if st.session_state.confirmed:
            st.session_state.pending = st.session_state.confirmed.copy()
    if st.button("Clear pending", key="clear_pending_btn"):
        st.session_state.pending = None
        st.session_state.nearby = []
    st.markdown("---")
    st.subheader("Weather")
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
                "precip_prob": hourly.get("precipitation_probability", []),
                "wind": hourly.get("wind_speed_10m", []),
            })
            st.markdown("Next hours")
            try:
                st.line_chart(df_hour.set_index("time")[["temperature"]].head(48))
                st.line_chart(df_hour.set_index("time")[["precip_prob"]].head(48))
            except Exception:
                pass
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
            st.markdown(f"Daily ({st.session_state['forecast_days']} days)")
            try:
                st.dataframe(df_daily.set_index("date").head(st.session_state["forecast_days"]))
            except Exception:
                pass
    else:
        st.info(I18N[lang]["no_weather"])
    st.markdown("---")
    alt_local = None
    if st.session_state.confirmed and "altitude_m" in st.session_state.confirmed:
        alt_local = st.session_state.confirmed.get("altitude_m")
    if st.session_state.pending and "altitude_m" in st.session_state.pending:
        alt_local = st.session_state.pending.get("altitude_m")
    if alt_local is not None:
        st.write(f"⛰ Altitude (OpenTopoData): {alt_local:.1f} m")
    if st.button(I18N[lang]["confirm"], key="sidebar_confirm_btn"):
        if st.session_state.pending:
            st.session_state.confirmed = st.session_state.pending.copy()
            st.session_state.pending = None
            label_city = st.session_state.confirmed.get("city")
            if label_city and label_city not in st.session_state.mru:
                st.session_state.mru.append(label_city)
    st.markdown("---")
    st.subheader(I18N[lang]["recent"])
    for r in st.session_state.mru[-8:][::-1]:
        if st.button(f"→ {r}", key=f"mru_{r}"):
            mask = cities_df["display_name"] == r
            if mask.any():
                rr = cities_df[mask].iloc[0]
                st.session_state.pending = {"lat": float(rr["lat"]), "lon": float(rr["lon"]), "city": r}
                st.session_state.weather = get_weather_open_meteo(rr["lat"], rr["lon"], days=st.session_state["forecast_days"])
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(rr["lat"], rr["lon"])

# -------------------------
# Map area
# -------------------------
active = st.session_state.pending if st.session_state.pending else st.session_state.confirmed
st.markdown("<div class='map-area'>", unsafe_allow_html=True)
st.subheader("🗺 Map — click to pick coordinates, drag marker to move, then Confirm")

if USE_FOLIUM and USE_STREAMLIT_FOLIUM:
    m = folium.Map(location=[active["lat"], active["lon"]], zoom_start=4, tiles=None, control_scale=True)
    if base_map.startswith("Hybrid"):
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery, © ESRI",
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
        folium.raster_layers.TileLayer(tiles=tpl, attr="NASA GIBS MODIS", name="MODIS LST", overlay=True, control=False, opacity=0.6, max_zoom=9).add_to(m)
    if st.session_state.nearby:
        cluster = MarkerCluster().add_to(m)
        for it in st.session_state.nearby:
            col = temp_to_hex(it.get("temp"))
            popup_html = f"{it['display_name']}<br>{it.get('dist_km', 0):.1f} km"
            folium.CircleMarker(location=(it["lat"], it["lon"]), radius=6, color=col, fill=True, fill_color=col, fill_opacity=0.9, popup=popup_html).add_to(cluster)
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
    marker = folium.Marker(location=[active["lat"], active["lon"]], draggable=True, tooltip="Drag marker to adjust coordinates", icon=folium.Icon(color="red", icon="map-marker"))
    if popup_brief:
        marker.add_child(folium.Popup(popup_brief, max_width=320))
    marker.add_to(m)
    folium.LayerControl(position="topright", collapsed=True).add_to(m)
    map_data = st_folium(m, width="100%", height=MAP_HEIGHT, returned_objects=["last_clicked", "last_marker", "last_marker_drag"])
    if map_data:
        clicked = map_data.get("last_clicked")
        if clicked:
            latc = clicked.get("lat"); lonc = clicked.get("lng")
            if latc is not None and lonc is not None:
                st.session_state.pending = {"lat": float(latc), "lon": float(lonc), "city": f"{latc:.5f},{lonc:.5f}"}
                if city_lats.size > 0:
                    d = haversine_array(latc, lonc, city_lats, city_lons)
                    idxs = np.argsort(d)[:NEAREST_SUGGESTIONS]
                    nearest = []
                    for i in idxs:
                        nearest.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
                    st.session_state.nearby = nearest
                st.session_state.weather = get_weather_open_meteo(latc, lonc, days=st.session_state["forecast_days"])
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(latc, lonc)
                st.session_state.map_used = True
        drag = map_data.get("last_marker_drag") or map_data.get("last_marker")
        if drag:
            latm = drag.get("lat"); lonm = drag.get("lng")
            if latm is not None and lonm is not None:
                st.session_state.pending = {"lat": float(latm), "lon": float(lonm), "city": f"{latm:.5f},{lonm:.5f}"}
                if city_lats.size > 0:
                    d = haversine_array(latm, lonm, city_lats, city_lons)
                    idxs = np.argsort(d)[:NEAREST_SUGGESTIONS]
                    nearest = []
                    for i in idxs:
                        nearest.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
                    st.session_state.nearby = nearest
                st.session_state.weather = get_weather_open_meteo(latm, lonm, days=st.session_state["forecast_days"])
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(latm, lonm)
                st.session_state.map_used = True

elif USE_PYDECK:
    esri = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    layers = [pdk.Layer("TileLayer", data=esri, tileSize=256)]
    pt = pd.DataFrame([{"lat": active["lat"], "lon": active["lon"]}])
    layers.append(pdk.Layer("ScatterplotLayer", data=pt, get_position='[lon, lat]', get_color='[255,0,0,255]', get_radius=120000))
    view = pdk.ViewState(latitude=active["lat"], longitude=active["lon"], zoom=4, pitch=30)
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style=None)
    st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
    st.info("Pydeck fallback: less interactive than folium. Install folium & streamlit-folium for marker drag/click features.")
else:
    st.error("No mapping libraries available. Install folium & streamlit-folium or pydeck.")

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# After map: quick summary + ML block
# -------------------------
left_col, right_col = st.columns([3, 1])
with left_col:
    if st.session_state.pending or st.session_state.confirmed:
        cur = st.session_state.pending if st.session_state.pending else st.session_state.confirmed
        label = I18N[lang]["pending"] if st.session_state.pending else I18N[lang]["confirmed"]
        st.markdown(f"**{label}**: {cur.get('city', 'Unknown')}")
        st.markdown(f"{I18N[lang]['coords_label']}: {cur['lat']:.6f}, {cur['lon']:.6f}")

    if st.session_state.nearby:
        st.markdown(f"### {I18N[lang]['nearest_label']}")
        for i, item in enumerate(st.session_state.nearby):
            btn_label = f"{item['display_name']} — {item['dist_km']:.1f} km"
            if st.button(btn_label, key=f"near_{i}"):
                st.session_state.pending = {"lat": item["lat"], "lon": item["lon"], "city": item["display_name"]}
                st.session_state.weather = get_weather_open_meteo(item["lat"], item["lon"], days=st.session_state["forecast_days"])
                st.session_state.pending["altitude_m"] = get_altitude_opentopo(item["lat"], item["lon"])
                st.session_state.nearby = []

    ca, cb, cc = st.columns([1, 1, 1])
    if ca.button(I18N[lang]["confirm"]):
        if st.session_state.pending:
            st.session_state.confirmed = st.session_state.pending.copy()
            st.session_state.pending = None
            label_city = st.session_state.confirmed.get("city")
            if label_city and label_city not in st.session_state.mru:
                st.session_state.mru.append(label_city)
    if cb.button(I18N[lang]["cancel"]):
        st.session_state.pending = None
        st.session_state.nearby = []
        st.session_state.weather = None
    if cc.button(I18N[lang]["nearby_quick"]):
        p = st.session_state.pending if st.session_state.pending else st.session_state.confirmed
        lat0 = p["lat"]; lon0 = p["lon"]
        if city_lats.size > 0:
            d = haversine_array(lat0, lon0, city_lats, city_lons)
            idxs = np.argsort(d)[:200] if len(d) > 0 else []
            nearby = []
            for i in idxs:
                nearby.append({"display_name": city_disp[i], "lat": float(city_lats[i]), "lon": float(city_lons[i]), "dist_km": float(d[i])})
            st.session_state.nearby = nearby
    st.markdown("---")

    if st.session_state.weather:
        st.markdown(f"### {I18N[lang]['next_48h']}")
        try:
            w = st.session_state.weather
            hourly = w.get("hourly", {})
            if hourly and "time" in hourly:
                df_hour = pd.DataFrame({
                    "time": pd.to_datetime(hourly.get("time", [])),
                    "temperature": hourly.get("temperature_2m", []),
                    "precip_prob": hourly.get("precipitation_probability", []),
                })
                st.line_chart(df_hour.set_index("time")[["temperature"]].head(48))
                st.dataframe(df_hour.head(12).set_index("time"))
                csv = df_hour.to_csv(index=False).encode("utf-8")
                st.download_button(I18N[lang]["download_csv"], data=csv, file_name="hourly.csv", mime="text/csv")
        except Exception:
            pass

    # --------------------
    # ML + heuristics block
    # --------------------
    if st.session_state.weather:
        aggregates = extract_simple_features_from_weather(st.session_state.weather)
        pred_resp = None
        if aggregates:
            pred_resp = call_backend_prediction(aggregates)
        heur = compute_heuristic_scores(st.session_state.weather)

        st.markdown("---")
        st.subheader("🔮 Weather risk — ML + Heuristics")
        hcol1, hcol2 = st.columns([1, 1.2])
        with hcol1:
            st.write("**Local heuristics (quick)**")
            heuristic_items = [
                ("very_hot","Very hot","High maximum temperature expected"),
                ("very_cold","Very cold","Very low minimum temperature"),
                ("very_windy","Very windy","Strong peak winds possible"),
                ("very_wet","Very wet","High precipitation probability"),
                ("very_uncomfortable","Very uncomfortable","Heat + humidity discomfort"),
            ]
            for key, label, desc in heuristic_items:
                score = heur.get(key, 0.0)
                st.markdown(f"**{label}** — {pct(score)}%")
                st.progress(pct(score))
                st.caption(desc)
            st.caption("Heuristics computed locally from Open-Meteo hourly aggregates. These are quick indicators, not the ML model's prediction.")
        with hcol2:
            st.write("**ML model response**")
            if pred_resp is None:
                st.info("ML: no aggregates or backend unavailable.")
            else:
                if isinstance(pred_resp, dict) and pred_resp.get("error"):
                    st.error(f"Prediction error (backend): {pred_resp.get('error')}")
                else:
                    pred_raw = pred_resp.get("prediction") if isinstance(pred_resp, dict) else pred_resp
                    probs = pred_resp.get("probs") if isinstance(pred_resp, dict) else None
                    note = pred_resp.get("note") if isinstance(pred_resp, dict) else None
                    original = pred_resp.get("original_features") if isinstance(pred_resp, dict) else None
                    used = pred_resp.get("used_features") if isinstance(pred_resp, dict) else None

                    display_pred = pred_raw
                    pred_is_num = False
                    try:
                        display_pred = float(pred_raw)
                        pred_is_num = True
                    except Exception:
                        pred_is_num = False

                    if pred_is_num:
                        score_val = round(display_pred, 3)
                        st.metric(label="Risk score (model)", value=f"{score_val}")
                        def interpret_numeric_pred(v):
                            try:
                                vv = float(v)
                            except Exception:
                                return "unknown"
                            if vv >= 2.0:
                                return "Very likely"
                            if vv >= 1.0:
                                return "Likely"
                            if vv >= 0.0:
                                return "Possible"
                            if vv < 0.0:
                                return "Unlikely"
                            return "unknown"
                        label_text = interpret_numeric_pred(display_pred)
                        if label_text == "Very likely":
                            st.warning(f"Interpreted: {label_text}")
                        elif label_text == "Likely":
                            st.info(f"Interpreted: {label_text}")
                        elif label_text == "Possible":
                            st.success(f"Interpreted: {label_text}")
                        else:
                            st.info(f"Interpreted: {label_text}")
                    else:
                        st.write("Prediction (raw):")
                        st.write(display_pred)

                    if probs:
                        try:
                            dfp = pd.DataFrame(probs)
                            st.markdown("**Probabilities:**")
                            st.dataframe(dfp)
                        except Exception:
                            st.write("Probabilities:")
                            st.write(probs)

                    if note:
                        st.warning(f"Feature adaptation: {note}")

                    # Friendly feature tables
                    FEATURE_DEFS = [
                        ("temp", "Max temp (°C)", "Maximum hourly temperature"),
                        ("hum", "Mean humidity (%)", "Average relative humidity"),
                        ("wind", "Max wind (m/s)", "Peak wind speed"),
                        ("precip", "Mean precip prob (%)", "Average precipitation probability"),
                        ("uv", "Mean UV index", "Average UV index"),
                    ]
                    def _features_to_df(vals):
                        rows = []
                        for i, v in enumerate(list(vals or [])):
                            name, label, desc = FEATURE_DEFS[i] if i < len(FEATURE_DEFS) else (f"f{i}", f"f{i}", "")
                            if name == "temp":
                                unit = "°C"
                            elif name in ("hum", "precip"):
                                unit = "%"
                            elif name == "wind":
                                unit = "m/s"
                            elif name == "uv":
                                unit = "index"
                            else:
                                unit = ""
                            try:
                                val = float(v)
                            except Exception:
                                val = v
                            rows.append({"feature": f"f{i}", "name": name, "label": label, "value": val, "unit": unit, "description": desc})
                        return pd.DataFrame(rows)
                    try:
                        df_orig = _features_to_df(original)
                        if not df_orig.empty:
                            st.markdown("**Original features sent (friendly):**")
                            st.dataframe(df_orig.set_index("feature")[ ["label","value","unit","description"] ])
                            legend_lines = [f"{r['feature']} → {r['label']}" for _, r in df_orig.iterrows()]
                            st.caption("Feature mapping: " + ", ".join(legend_lines))
                    except Exception:
                        if original:
                            st.write("Original features:")
                            st.write(original)
                    if used is not None:
                        try:
                            df_used = _features_to_df(used)
                            if not df_used.empty:
                                st.markdown("**Features used by model (friendly):**")
                                st.dataframe(df_used.set_index("feature")[ ["label","value","unit","description"] ])
                        except Exception:
                            st.write("Used features:")
                            st.write(used)
                    with st.expander("Show full backend response (debug)"):
                        st.json(pred_resp)
with right_col:
    st.markdown("### Map quick")
    try:
        st.write(f"Base map: {base_map}")
        st.write(f"MODIS date: {tile_date.isoformat()}")
        st.write("MODIS overlay: ON" if show_modis else "MODIS overlay: OFF")
    except Exception:
        pass
    st.markdown("---")
    st.write("Tips:")
    st.write("- Drag the marker to refine coordinates.")
    st.write("- Click the map to pick a point.")
    st.write("- Use 'Nearby (quick)' to expand nearest city list.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Local cities only (data/cities.json). For best draggable marker experience install folium & streamlit-folium.")

# End of file

