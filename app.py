
import os
import joblib
import pandas as pd
import requests
import numpy as np
import gradio as gr


# =======================================================
# ⚙️ CONFIGURATION
# =======================================================

ML_FEATURES = [
    "rainfall_3day_mm",
    "rainfall_7day_mm",
    "slope_degrees",
    "soil_moisture_m3_m3",
    "elevation_m"
]

API_TIMEOUT = 30


# =======================================================
# 🤖 LOAD PRODUCTION MODEL
# =======================================================

MODEL_PATH = "landslide_gradient_boosting_model.pkl"

model = joblib.load(MODEL_PATH)


# =======================================================
# 🌧️ LIVE WEATHER
# =======================================================

def get_live_weather(lat, lon):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,"
        "precipitation,soil_moisture_0_to_7cm"
        "&daily=precipitation_sum"
        "&past_days=7&forecast_days=0&timezone=auto"
    )

    response = requests.get(url, timeout=API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"Open-Meteo API Error {response.status_code}: {response.text}")
        
    data = response.json()

    rainfall = data["daily"]["precipitation_sum"]

    rainfall = [
        0 if value is None else value
        for value in rainfall
    ]

    rainfall_7day = sum(rainfall[-7:])
    rainfall_3day = sum(rainfall[-3:])

    current = data["current"]

    return {
        "rainfall_3day_mm": round(float(rainfall_3day), 2),
        "rainfall_7day_mm": round(float(rainfall_7day), 2),
        "soil_moisture_m3_m3": float(
            current["soil_moisture_0_to_7cm"]
        ),
        "temperature_c": float(
            current["temperature_2m"]
        ),
        "humidity_percent": float(
            current["relative_humidity_2m"]
        ),
        "current_precipitation_mm": float(
            current["precipitation"]
        )
    }


# =======================================================
# ⛰️ TERRAIN
# =======================================================

def get_terrain_features(lat, lon):

    points = [
        (lat, lon),
        (lat + 0.01, lon),
        (lat - 0.01, lon),
        (lat, lon + 0.01),
        (lat, lon - 0.01)
    ]

    elevations = []

    for point_lat, point_lon in points:

        url = (
            "https://api.open-meteo.com/v1/elevation"
            f"?latitude={point_lat}"
            f"&longitude={point_lon}"
        )

        response = requests.get(
            url,
            timeout=API_TIMEOUT
        )

        response.raise_for_status()

        data = response.json()

        elevations.append(
            float(data["elevation"][0])
        )

    center = elevations[0]
    north = elevations[1]
    south = elevations[2]
    east = elevations[3]
    west = elevations[4]

    dy = 2224.0
    dx = 2015.0

    dz_dx = (east - west) / (2 * dx)
    dz_dy = (north - south) / (2 * dy)

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx**2 + dz_dy**2
            )
        )
    )

    return round(float(slope), 2), round(center, 2)
# =======================================================
# 🚨 RISK CLASSIFICATION
# =======================================================

def classify_risk(risk_score):

    if risk_score >= 70:
        return "HIGH RISK"

    elif risk_score >= 50:
        return "MODERATE RISK"

    else:
        return "LOWER RISK"


# =======================================================
# 🤖 COMPLETE LIVE PREDICTION
# =======================================================

def predict_live_landslide_risk(lat, lon):

    weather = get_live_weather(lat, lon)

    slope, elevation = get_terrain_features(
        lat,
        lon
    )

    features = pd.DataFrame([{
        "rainfall_3day_mm":
            weather["rainfall_3day_mm"],

        "rainfall_7day_mm":
            weather["rainfall_7day_mm"],

        "slope_degrees":
            slope,

        "soil_moisture_m3_m3":
            weather["soil_moisture_m3_m3"],

        "elevation_m":
            elevation
    }])

    risk_score = (
        model.predict_proba(
            features[ML_FEATURES]
        )[0][1] * 100
    )

    risk_score = round(
        float(risk_score),
        2
    )

    risk_level = classify_risk(
        risk_score
    )

    return {
        "rainfall_3day_mm":
            weather["rainfall_3day_mm"],

        "rainfall_7day_mm":
            weather["rainfall_7day_mm"],

        "soil_moisture_m3_m3":
            weather["soil_moisture_m3_m3"],

        "temperature_c":
            weather["temperature_c"],

        "humidity_percent":
            weather["humidity_percent"],

        "current_precipitation_mm":
            weather["current_precipitation_mm"],

        "slope_degrees":
            slope,

        "elevation_m":
            elevation,

        "ai_risk_score":
            risk_score,

        "risk_level":
            risk_level
    }


# =======================================================
# =======================================================
# 🌍 SUPPORTED STATES
# =======================================================

SUPPORTED_STATES = [
    "Assam",
    "Arunachal Pradesh",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]


# =======================================================
# 📍 DETECT STATE FROM COORDINATES
# =======================================================

def get_state_from_coordinates(lat, lon):

    url = (
        "https://nominatim.openstreetmap.org/reverse"
        f"?lat={lat}"
        f"&lon={lon}"
        "&format=jsonv2"
        "&addressdetails=1"
        "&zoom=5"
    )

    headers = {
        "User-Agent":
            "SIH26001-Landslide-Early-Warning/1.0"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=API_TIMEOUT
    )

    if response.status_code != 200:
        raise Exception(
            f"Location detection error: "
            f"{response.status_code}"
        )

    data = response.json()

    address = data.get("address", {})

    state = address.get("state")

    return state


# =======================================================
# 🎨 GRADIO FUNCTION
# =======================================================

def gradio_prediction(lat, lon):

    try:

        if lat is None or lon is None:
            raise Exception(
                "Please select a location first."
            )

        lat = float(lat)
        lon = float(lon)


        # ------------------------------------------------
        # DETECT STATE
        # ------------------------------------------------

        detected_state = get_state_from_coordinates(
            lat,
            lon
        )

        print(
            f"Selected coordinates: "
            f"{lat}, {lon}"
        )

        print(
            f"Detected state: "
            f"{detected_state}"
        )


        # ------------------------------------------------
        # BLOCK UNSUPPORTED REGION
        # ------------------------------------------------

        if detected_state not in SUPPORTED_STATES:

            return (
                "NOT SUPPORTED",
                "N/A",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                (
                    "⚠️ AI prediction is currently "
                    "available only for Northeast India. "
                    f"Selected state: "
                    f"{detected_state or 'Unknown'}. "
                    "The current model is not validated "
                    "for this region."
                )
            )


        # ------------------------------------------------
        # RUN EXISTING AI MODEL
        # ------------------------------------------------

        result = predict_live_landslide_risk(
            lat,
            lon
        )


        risk_level = result["risk_level"]

        risk_score = result["ai_risk_score"]


        # ------------------------------------------------
        # WARNING
        # ------------------------------------------------

        if risk_level == "HIGH RISK":

            warning = (
                "🚨 High landslide risk detected. "
                "Exercise caution and monitor "
                "local warnings."
            )

        elif risk_level == "MODERATE RISK":

            warning = (
                "⚠️ Moderate landslide risk. "
                "Stay alert and monitor "
                "weather conditions."
            )

        else:

            warning = (
                "✅ Lower landslide risk under "
                "current conditions."
            )


        # ------------------------------------------------
        # RETURN RESULTS
        # ------------------------------------------------

        return (
            risk_level,
            f"{risk_score}%",
            result["rainfall_3day_mm"],
            result["rainfall_7day_mm"],
            result["soil_moisture_m3_m3"],
            result["slope_degrees"],
            result["elevation_m"],
            result["temperature_c"],
            result["humidity_percent"],
            warning
        )


    except Exception as e:

        return (
            "ERROR",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            f"❌ Error: {str(e)}"
        )
    # ========================================================
    # 📚 SYSTEM INFORMATION
    # ========================================================

    with gr.Row():

        with gr.Column(elem_classes="info-panel"):

            gr.Markdown("""
### 📚 Data Sources

🌧️ **Weather:** Open-Meteo API  
⛰️ **Elevation:** Open-Meteo Elevation API  
🗺️ **Map:** OpenStreetMap  
🛰️ **Historical Landslides:** NASA Global Landslide Catalog  
🤖 **AI Model:** Gradient Boosting Classifier
""")

        with gr.Column(elem_classes="info-panel"):

            gr.Markdown("""
### 🌍 Supported Regions

**Northeast India**

🇮🇳 Assam  
🇮🇳 Arunachal Pradesh  
🇮🇳 Manipur  
🇮🇳 Meghalaya  
🇮🇳 Mizoram  
🇮🇳 Nagaland  
🇮🇳 Sikkim  
🇮🇳 Tripura
""")
# 
# ============================================================
# 🗺️ FINAL DASHBOARD UI
# STATE SEARCH + MAP + AUTO COORDINATES + DAY/NIGHT THEME
# ============================================================

STATE_COORDINATES = {
    "Assam": (26.1445, 91.7362),              # Guwahati
    "Arunachal Pradesh": (27.0844, 93.6053), # Itanagar
    "Manipur": (24.8170, 93.9368),            # Imphal
    "Meghalaya": (25.5788, 91.8933),          # Shillong
    "Mizoram": (23.7271, 92.7176),            # Aizawl
    "Nagaland": (25.6751, 94.1086),           # Kohima
    "Sikkim": (27.3389, 88.6065),             # Sikkim
    "Tripura": (23.8315, 91.2868)             # Agartala
}


def select_state(state):

    if state in STATE_COORDINATES:
        return STATE_COORDINATES[state]

    return 27.3389, 88.6065


# ============================================================
# 🎨 CUSTOM DASHBOARD STYLE
# ============================================================

CUSTOM_CSS = """

/* ============================================================
   🌗 AUTOMATIC DAY / NIGHT THEME
   ============================================================ */


/* ============================================================
   ☀️ DAY MODE
   6 AM - 6 PM
   ============================================================ */

body.day-mode {
    background: #eef7f5 !important;
    transition: background 0.8s ease;
}

body.day-mode .gradio-container {
    background: #eef7f5 !important;
    color: #173b36 !important;
}


/* Day mode cards */

body.day-mode .glass-card,
body.day-mode .risk-card {
    background: rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(0,121,107,0.22);
    box-shadow: 0 8px 25px rgba(0,80,70,0.10);
}


/* Day mode inputs */

body.day-mode input,
body.day-mode textarea,
body.day-mode button {
    border-color: rgba(0,121,107,0.25) !important;
}


/* ============================================================
   🌌 NIGHT MODE
   6 PM - 6 AM
   ============================================================ */

body.night-mode {
    background:
        radial-gradient(
            circle at 20% 10%,
            rgba(0,150,136,0.12),
            transparent 35%
        ),
        #071413 !important;

    transition: background 0.8s ease;
}

body.night-mode .gradio-container {
    background: transparent !important;
    color: #e8fffb !important;
}


/* Night cards */

body.night-mode .glass-card,
body.night-mode .risk-card {
    background: rgba(11,35,33,0.82) !important;

    border: 1px solid rgba(0,188,174,0.25);

    box-shadow:
        0 8px 30px rgba(0,0,0,0.35),
        0 0 20px rgba(0,150,136,0.06);
}


/* Night inputs */

body.night-mode input,
body.night-mode textarea {
    background: #102321 !important;
    color: #e8fffb !important;
    border-color: rgba(0,188,174,0.25) !important;
}


/* ============================================================
   🚨 HEADER
   ============================================================ */

.dashboard-header {
    padding: 30px;
    border-radius: 24px;
    margin-bottom: 25px;

    background:
        linear-gradient(
            135deg,
            #008f83,
            #00695c
        );

    color: white;

    box-shadow:
        0 15px 35px rgba(0,80,70,0.22);
}

.dashboard-header h1 {
    font-size: 34px;
    margin: 0;
}

.dashboard-header p {
    font-size: 16px;
    margin-top: 10px;
}


/* ============================================================
   🌌 NIGHT HEADER
   ============================================================ */

body.night-mode .dashboard-header {

    background:
        linear-gradient(
            135deg,
            #063f3b,
            #071f1e,
            #052b29
        );

    box-shadow:
        0 15px 40px rgba(0,0,0,0.45),
        0 0 30px rgba(0,188,174,0.08);
}


/* ============================================================
   📍 SECTION HEADINGS
   ============================================================ */

.section-title {
    font-size: 22px;
    font-weight: 700;

    margin-top: 24px;
    margin-bottom: 12px;
}


/* ============================================================
   🗺️ MAP
   ============================================================ */

#landslide-map {

    height: 500px !important;
    width: 100% !important;

    border-radius: 20px;
    overflow: hidden;

    border: 2px solid rgba(0,137,123,0.40);

    box-shadow:
        0 12px 30px rgba(0,0,0,0.18);
}


/* ============================================================
   🌌 NIGHT MAP BORDER
   ============================================================ */

body.night-mode #landslide-map {

    border-color:
        rgba(0,188,174,0.45);

    box-shadow:
        0 12px 35px rgba(0,0,0,0.45),
        0 0 20px rgba(0,188,174,0.08);
}


/* ============================================================
   🔍 ANALYZE BUTTON
   ============================================================ */

.analyze-button {

    border-radius: 13px !important;

    font-weight: 700 !important;

    min-height: 48px !important;
}


/* ============================================================
   🌊 NIGHT BUTTON
   ============================================================ */

body.night-mode .analyze-button {

    box-shadow:
        0 0 15px rgba(0,188,174,0.12);
}


/* ============================================================
   📱 RESPONSIVE
   ============================================================ */

@media (max-width: 800px) {

    .dashboard-header h1 {
        font-size: 27px;
    }

    #landslide-map {
        height: 400px !important;
    }

}

/* ==========================================================
   🤖 AI RISK ASSESSMENT - PREMIUM CARD
   ========================================================== */

.risk-card {
    border-radius: 18px !important;
    border: 1px solid rgba(20, 184, 166, 0.35) !important;
    overflow: hidden !important;
}

.risk-card input,
.risk-card textarea {
    font-size: 20px !important;
    font-weight: 700 !important;
    min-height: 55px !important;
}

/* 🌙 NIGHT MODE */

body.night-mode .risk-card {
    background: rgba(0, 35, 32, 0.75) !important;
    box-shadow:
        0 0 25px rgba(20, 184, 166, 0.12),
        inset 0 0 20px rgba(0, 255, 200, 0.03) !important;
}

body.night-mode .risk-card input,
body.night-mode .risk-card textarea {
    background: rgba(0, 20, 18, 0.85) !important;
    color: #8fffe8 !important;
    border-color: rgba(20, 184, 166, 0.35) !important;
}

/* ☀️ DAY MODE */

body.day-mode .risk-card {
    background: rgba(255, 255, 255, 0.72) !important;
    box-shadow: 0 8px 25px rgba(13, 148, 136, 0.12) !important;
}

body.day-mode .risk-card input,
body.day-mode .risk-card textarea {
    background: rgba(255, 255, 255, 0.9) !important;
    color: #064e49 !important;
    border-color: rgba(20, 184, 166, 0.35) !important;
}
/* ==========================================================
   🌌 FINAL SIH DASHBOARD POLISH
   ========================================================== */

/* ---------- COMMON METRIC CARDS ---------- */

/* ==========================================================
   🌧️ ENVIRONMENTAL / TERRAIN / WEATHER
   ========================================================== */

body.day-mode input,
body.day-mode textarea {
    border-radius: 12px !important;
}

body.night-mode input,
body.night-mode textarea {
    border-radius: 12px !important;
}


/* ---------- SECTION SPACING ---------- */

.section-title {
    letter-spacing: 0.2px;
}


/* ==========================================================
   🌧️ ENVIRONMENTAL CONDITIONS
   ========================================================== */

body.day-mode .section-title {
    color: #064e49 !important;
}

body.night-mode .section-title {
    color: #d9fffa !important;
}


/* Metric value emphasis */

body.day-mode input[type="number"],
body.night-mode input[type="number"] {
    font-size: 18px !important;
    font-weight: 700 !important;
}


/* ==========================================================
   📍 LOCATION PANEL
   ========================================================== */

body.night-mode #latitude_input input,
body.night-mode #longitude_input input {
    background: #102321 !important;
    color: #8fffe8 !important;
}

body.day-mode #latitude_input input,
body.day-mode #longitude_input input {
    background: rgba(255,255,255,0.88) !important;
    color: #064e49 !important;
}


/* ==========================================================
   🔍 STATE SEARCH
   ========================================================== */

body.night-mode .gradio-dropdown,
body.night-mode [role="listbox"] {
    background: #102321 !important;
    color: #e8fffb !important;
}

body.day-mode .gradio-dropdown {
    background: rgba(255,255,255,0.90) !important;
}


/* ==========================================================
   🔥 ANALYZE BUTTON
   ========================================================== */

.analyze-button {
    margin-top: 8px !important;
    border-radius: 14px !important;
    font-size: 16px !important;
    font-weight: 800 !important;
    letter-spacing: 0.2px;
    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.analyze-button:hover {
    transform: translateY(-2px);
}

body.night-mode .analyze-button {
    box-shadow:
        0 0 20px rgba(0,255,210,0.18),
        0 8px 20px rgba(0,0,0,0.30);
}


/* ==========================================================
   ⚠️ SYSTEM WARNING
   ========================================================== */

body.night-mode .gradio-textbox {
    border-radius: 16px !important;
}

body.night-mode textarea {
    background: rgba(7,25,23,0.85) !important;
}


/* ==========================================================
   🗺️ MAP GLOW
   ========================================================== */

body.night-mode #landslide-map {
    box-shadow:
        0 12px 35px rgba(0,0,0,0.50),
        0 0 35px rgba(0,188,174,0.12) !important;
}

body.day-mode #landslide-map {
    box-shadow:
        0 12px 30px rgba(13,148,136,0.15) !important;
}


/* ==========================================================
   🌌 NIGHT AURORA EFFECT
   ========================================================== */

body.night-mode {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(0,255,210,0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(0,180,160,0.08),
            transparent 30%
        ),
        #071413 !important;
}


/* ==========================================================
   ☀️ DAY NORTHERN LIGHTS
   ========================================================== */

body.day-mode {
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(45,212,191,0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(34,211,238,0.14),
            transparent 30%
        ),
        #eaf9f6 !important;
}


/* ==========================================================
   📱 MOBILE POLISH
   ========================================================== */

@media (max-width: 800px) {

    .dashboard-header {
        padding: 22px !important;
        border-radius: 20px !important;
    }

    .dashboard-header h1 {
        font-size: 26px !important;
        line-height: 1.2;
    }

    .dashboard-header p {
        font-size: 14px !important;
    }

    #landslide-map {
        height: 360px !important;
        border-radius: 16px !important;
    }

    .section-title {
        font-size: 20px !important;
    }

    .risk-card input {
        font-size: 18px !important;
    }

}


/* ==========================================================
   ✨ SMOOTH UI
   ========================================================== */

.gradio-container,
.dashboard-header,
.risk-card,
.analyze-button,
#landslide-map {
    transition:
        background 0.5s ease,
        border-color 0.5s ease,
        box-shadow 0.5s ease,
        transform 0.2s ease;
}
/* ==========================================================
   🌌 PREMIUM METRIC CARDS
   ========================================================== */

.metric-card {
    border-radius: 20px !important;
    padding: 4px !important;
    min-height: 115px !important;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease !important;
}


/* ☀️ DAY */

body.day-mode .metric-card {
    background: rgba(255,255,255,0.82) !important;

    border: 1px solid rgba(13,148,136,0.22) !important;

    box-shadow:
        0 8px 25px rgba(13,148,136,0.10) !important;
}


/* 🌙 NIGHT */

body.night-mode .metric-card {
    background:
        linear-gradient(
            145deg,
            rgba(9,43,40,0.92),
            rgba(5,28,27,0.88)
        ) !important;

    border: 1px solid rgba(20,184,166,0.30) !important;

    box-shadow:
        0 8px 28px rgba(0,0,0,0.38),
        0 0 22px rgba(20,184,166,0.07) !important;
}


/* Hover */

.metric-card:hover {
    transform: translateY(-4px);
}

body.night-mode .metric-card:hover {
    border-color: rgba(45,212,191,0.55) !important;

    box-shadow:
        0 12px 32px rgba(0,0,0,0.42),
        0 0 28px rgba(20,184,166,0.14) !important;
}


/* Metric values */

.metric-card input {
    font-size: 22px !important;
    font-weight: 800 !important;
    text-align: center !important;
}


/* Night metric values */

body.night-mode .metric-card input {
    color: #8fffe8 !important;
    background: rgba(3,25,23,0.65) !important;
    border-color: rgba(20,184,166,0.18) !important;
}


/* Day metric values */

body.day-mode .metric-card input {
    color: #075e57 !important;
    background: rgba(255,255,255,0.72) !important;
}


/* Metric labels */

.metric-card label {
    font-weight: 700 !important;
}


/* ==========================================================
   🚨 PREMIUM RISK PANELS
   ========================================================== */

.risk-panel {
    border-radius: 22px !important;
    padding: 5px !important;

    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease !important;
}


/* Day risk */

body.day-mode .risk-panel {
    background: rgba(255,255,255,0.86) !important;

    border: 1px solid rgba(13,148,136,0.28) !important;

    box-shadow:
        0 10px 30px rgba(13,148,136,0.12) !important;
}


/* Night risk */

body.night-mode .risk-panel {
    background:
        linear-gradient(
            145deg,
            rgba(5,48,44,0.95),
            rgba(3,27,26,0.92)
        ) !important;

    border: 1px solid rgba(45,212,191,0.32) !important;

    box-shadow:
        0 12px 35px rgba(0,0,0,0.42),
        0 0 30px rgba(20,184,166,0.10) !important;
}


/* Risk value */

.risk-value input {
    font-size: 25px !important;
    font-weight: 900 !important;
    min-height: 62px !important;
    text-align: center !important;
}


body.night-mode .risk-value input {
    color: #8fffe8 !important;
    background: rgba(0,22,20,0.82) !important;
}


body.day-mode .risk-value input {
    color: #075e57 !important;
    background: rgba(255,255,255,0.85) !important;
}


/* ==========================================================
   ⚠️ WARNING PANEL
   ========================================================== */

.warning-panel {
    margin-top: 14px !important;
    border-radius: 18px !important;

    border: 1px solid rgba(245,158,11,0.30) !important;
}


body.night-mode .warning-panel {
    background: rgba(48,34,8,0.55) !important;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.28),
        0 0 18px rgba(245,158,11,0.05) !important;
}


body.day-mode .warning-panel {
    background: rgba(255,251,235,0.90) !important;
}


/* ==========================================================
   📊 SECTION TITLES
   ========================================================== */

.section-title {
    margin-top: 30px !important;
    margin-bottom: 15px !important;

    font-size: 23px !important;
    font-weight: 800 !important;
}


/* ==========================================================
   📱 MOBILE
   ========================================================== */

@media (max-width: 800px) {

    .metric-card {
        min-height: 105px !important;
    }

    .metric-card input {
        font-size: 19px !important;
    }

    .risk-value input {
        font-size: 21px !important;
    }
}
/* ==========================================================
   📍 PREMIUM LOCATION PANEL
   ========================================================== */

.location-panel {
    border-radius: 22px !important;
    padding: 18px !important;
    transition:
        transform 0.25s ease,
        box-shadow 0.25s ease,
        border-color 0.25s ease !important;
}


/* ☀️ DAY */

body.day-mode .location-panel {
    background: rgba(255,255,255,0.78) !important;
    border: 1px solid rgba(13,148,136,0.22) !important;

    box-shadow:
        0 10px 30px rgba(13,148,136,0.10) !important;
}


/* 🌙 NIGHT */

body.night-mode .location-panel {
    background:
        linear-gradient(
            145deg,
            rgba(8,42,39,0.90),
            rgba(4,27,26,0.88)
        ) !important;

    border: 1px solid rgba(20,184,166,0.28) !important;

    box-shadow:
        0 12px 32px rgba(0,0,0,0.40),
        0 0 25px rgba(20,184,166,0.07) !important;
}


/* Location inputs */

.location-panel input {
    border-radius: 12px !important;
    font-weight: 700 !important;
}


/* Night */

body.night-mode .location-panel input {
    background: rgba(3,25,23,0.75) !important;
    color: #8fffe8 !important;
}


/* Day */

body.day-mode .location-panel input {
    background: rgba(255,255,255,0.85) !important;
    color: #075e57 !important;
}


/* ==========================================================
   🗺️ MAP PREMIUM FRAME
   ========================================================== */

body.night-mode #landslide-map {
    border: 2px solid rgba(45,212,191,0.42) !important;

    box-shadow:
        0 15px 40px rgba(0,0,0,0.45),
        0 0 30px rgba(20,184,166,0.10) !important;
}

body.day-mode #landslide-map {
    border: 2px solid rgba(13,148,136,0.30) !important;

    box-shadow:
        0 12px 30px rgba(13,148,136,0.14) !important;
}


/* ==========================================================
   📚 INFO SECTION
   ========================================================== */

.info-panel {
    margin-top: 30px !important;
    padding: 22px !important;
    border-radius: 20px !important;
}


body.night-mode .info-panel {
    background: rgba(7,35,32,0.78) !important;

    border: 1px solid rgba(20,184,166,0.22) !important;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.30) !important;

    color: #d9fffa !important;
}


body.day-mode .info-panel {
    background: rgba(255,255,255,0.82) !important;

    border: 1px solid rgba(13,148,136,0.18) !important;

    box-shadow:
        0 8px 25px rgba(13,148,136,0.08) !important;

    color: #24514b !important;
}


/* ==========================================================
   🏆 SIH FOOTER
   ========================================================== */

.dashboard-footer {
    margin-top: 35px !important;
    padding: 22px 25px !important;

    border-radius: 20px !important;

    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;

    gap: 20px !important;
    flex-wrap: wrap !important;

    font-size: 13px !important;
}


body.night-mode .dashboard-footer {
    background:
        linear-gradient(
            135deg,
            rgba(5,40,37,0.92),
            rgba(3,24,23,0.92)
        ) !important;

    border: 1px solid rgba(20,184,166,0.20) !important;

    color: #b9fff5 !important;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.30) !important;
}


body.day-mode .dashboard-footer {
    background: rgba(255,255,255,0.85) !important;

    border: 1px solid rgba(13,148,136,0.16) !important;

    color: #24514b !important;
}


/* Footer mobile */

@media (max-width: 800px) {

    .location-panel {
        padding: 14px !important;
    }

    .dashboard-footer {
        text-align: center !important;
        justify-content: center !important;
    }

}
/* ==========================================================
   🌌 NORTHERN LIGHTS / AURORA BACKGROUND
   ========================================================== */

/* ==========================================================
   🌌 PERMANENT NORTHERN LIGHTS + MOUNTAIN SILHOUETTE
   NO DAY/NIGHT THEME SWITCH
   ========================================================== */

/* ---------- MAIN DARK TEAL BACKGROUND ---------- */

body {
    background:
        radial-gradient(
            ellipse at 50% 15%,
            rgba(0, 120, 110, 0.18),
            transparent 45%
        ),
        linear-gradient(
            180deg,
            #061c1b 0%,
            #071413 55%,
            #020b0b 100%
        ) !important;

    min-height: 100vh;
    overflow-x: hidden;
}


/* ---------- KEEP GRADIO TRANSPARENT ---------- */

.gradio-container {
    position: relative !important;
    z-index: 2 !important;

    background: transparent !important;
}


/* ==========================================================
   🌌 FLOWING NORTHERN LIGHTS
   ========================================================== */

body::before {
    content: "";

    position: fixed;

    top: -10%;
    left: -15%;

    width: 130%;
    height: 75%;

    pointer-events: none;

    z-index: 0;

    background:
        radial-gradient(
            ellipse 35% 55% at 15% 55%,
            rgba(0, 255, 190, 0.22),
            transparent 65%
        ),

        radial-gradient(
            ellipse 30% 60% at 38% 35%,
            rgba(0, 220, 180, 0.20),
            transparent 65%
        ),

        radial-gradient(
            ellipse 35% 55% at 62% 50%,
            rgba(0, 190, 220, 0.18),
            transparent 65%
        ),

        radial-gradient(
            ellipse 30% 50% at 85% 35%,
            rgba(0, 255, 170, 0.16),
            transparent 65%
        );

    filter: blur(28px);

    transform: rotate(-5deg);

    animation:
        auroraFlow 12s ease-in-out infinite alternate;
}


/* ==========================================================
   ✨ AURORA LIGHT RIBBON
   ========================================================== */

body::after {
    content: "";

    position: fixed;

    top: -5%;
    left: -20%;

    width: 140%;
    height: 65%;

    pointer-events: none;

    z-index: 0;

    background:
        linear-gradient(
            110deg,
            transparent 10%,
            rgba(0, 255, 190, 0.03) 25%,
            rgba(0, 255, 200, 0.14) 38%,
            rgba(0, 220, 190, 0.20) 48%,
            rgba(0, 200, 220, 0.12) 58%,
            rgba(0, 255, 190, 0.05) 70%,
            transparent 85%
        );

    filter: blur(22px);

    transform: skewX(-12deg) rotate(-4deg);

    animation:
        auroraWave 10s ease-in-out infinite alternate;
}


/* ==========================================================
   🌌 AURORA ANIMATION
   ========================================================== */

@keyframes auroraFlow {

    0% {
        transform:
            translateX(-4%)
            rotate(-5deg)
            scale(1);
    }

    50% {
        transform:
            translateX(3%)
            rotate(-2deg)
            scale(1.05);
    }

    100% {
        transform:
            translateX(-1%)
            rotate(-6deg)
            scale(1.08);
    }
}


@keyframes auroraWave {

    0% {
        transform:
            translateX(-5%)
            skewX(-12deg)
            rotate(-4deg);
    }

    50% {
        transform:
            translateX(5%)
            skewX(-7deg)
            rotate(-2deg);
    }

    100% {
        transform:
            translateX(-2%)
            skewX(-15deg)
            rotate(-5deg);
    }
}


/* ==========================================================
   🏔️ MOUNTAIN SILHOUETTE
   ========================================================== */

.dashboard-footer::before {
    content: "";

    position: absolute;

    left: 0;
    right: 0;

    bottom: 100%;

    height: 170px;

    pointer-events: none;

    background:
        linear-gradient(
            135deg,
            transparent 0 18%,
            #031211 18% 30%,
            transparent 30% 34%,
            #041715 34% 48%,
            transparent 48% 52%,
            #020d0d 52% 66%,
            transparent 66% 70%,
            #031211 70% 84%,
            transparent 84%
        );

    clip-path: polygon(
        0 100%,
        0 72%,
        8% 55%,
        15% 78%,
        23% 38%,
        31% 70%,
        40% 48%,
        48% 75%,
        58% 35%,
        67% 68%,
        76% 45%,
        85% 72%,
        93% 50%,
        100% 70%,
        100% 100%
    );

    opacity: 0.95;
}


/* Footer becomes mountain anchor */

.dashboard-footer {
    position: relative !important;
}


/* ==========================================================
   🌲 EXTRA DARK FOREGROUND
   ========================================================== */

body .gradio-container::after {
    content: "";

    position: fixed;

    left: 0;
    right: 0;
    bottom: 0;

    height: 90px;

    pointer-events: none;

    z-index: -1;

    background:
        linear-gradient(
            to top,
            rgba(1, 10, 10, 0.95),
            transparent
        );
}


/* ==========================================================
   📱 MOBILE AURORA
   ========================================================== */

@media (max-width: 800px) {

    body::before {
        height: 60%;
        filter: blur(24px);
    }

    body::after {
        height: 50%;
        filter: blur(18px);
    }

}
"""

# ============================================================
# 🚨 DASHBOARD
# ============================================================

with gr.Blocks(
    title="Landslide Early Warning System",
    css=CUSTOM_CSS
) as demo:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML("""
<div class="dashboard-header">

    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:25px;
        flex-wrap:wrap;
    ">

        <div>
            <h1 style="margin:0;">
                🏔️ Landslide Early Warning System
            </h1>

            <p style="
                margin:8px 0 0;
                font-size:16px;
            ">
                AI-Based Localized Landslide Risk Assessment for Northeast India
            </p>

            <div style="
                margin-top:14px;
                font-size:14px;
                display:flex;
                gap:18px;
                flex-wrap:wrap;
            ">
                <span>🌧️ Rainfall</span>
                <span>💧 Soil Moisture</span>
                <span>⛰️ Terrain</span>
                <span>🛰️ Environmental Data</span>
                <span>🤖 AI Risk Assessment</span>
            </div>
        </div>

        <div style="
            padding:14px 18px;
            border-radius:16px;
            background:rgba(0,0,0,0.16);
            border:1px solid rgba(255,255,255,0.20);
            text-align:center;
            min-width:150px;
        ">
            <div style="
                font-size:13px;
                opacity:0.85;
            ">
                🌗 AUTOMATIC THEME
            </div>

            <div style="
                margin-top:5px;
                font-size:17px;
                font-weight:700;
            ">
                Day / Night
            </div>

            <div style="
                margin-top:4px;
                font-size:12px;
                opacity:0.75;
            ">
                6 AM – 6 PM
            </div>
        </div>

    </div>

</div>
""")

    # ========================================================
    # 📍 LOCATION SECTION
    # ========================================================

    gr.Markdown(
        "## 📍 Select Location",
        elem_classes="section-title"
    )

    with gr.Row():

        # ----------------------------------------------------
        # LOCATION SEARCH
        # ----------------------------------------------------

        with gr.Column(scale=1):

            state_selector = gr.Dropdown(
                choices=list(STATE_COORDINATES.keys()),
                value="Sikkim",
                label="🔍 Search Northeast State",
                info="Only states covered by the current AI model.",
                allow_custom_value=False,
                filterable=True
            )

            latitude = gr.Number(
                label="Latitude",
                value=27.3389,
                precision=6,
                elem_id="latitude_input"
            )

            longitude = gr.Number(
                label="Longitude",
                value=88.6065,
                precision=6,
                elem_id="longitude_input"
            )

            gr.Markdown("""
            💡 **How to select a location**

            • Search a Northeast state above  
            • Or click directly on the map  
            • Coordinates update automatically
            """)

            predict_button = gr.Button(
                "🔍 Analyze Landslide Risk",
                variant="primary",
                elem_classes="analyze-button"
            )


        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        with gr.Column(scale=2):

            map_html = gr.HTML(
                html_template="""

                <div id="landslide-map"></div>

                """,

                head="""

                <link
                    rel="stylesheet"
                    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
                />

                <script
                    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
                </script>

                """,

                js_on_load="""

                const mapContainer =
                    element.querySelector("#landslide-map");

                if (!mapContainer) {
                    console.error("Map container not found");
                    return;
                }


                // ------------------------------------------------
                // MAP
                // ------------------------------------------------

                const map = L.map(
                    mapContainer
                ).setView(
                    [27.3389, 88.6065],
                    7
                );


                // ------------------------------------------------
                // OPEN STREET MAP
                // ------------------------------------------------

                L.tileLayer(
                    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    {
                        maxZoom: 19,
                        attribution:
                            "&copy; OpenStreetMap contributors"
                    }
                ).addTo(map);


                // ------------------------------------------------
                // MARKER
                // ------------------------------------------------

                let marker = L.marker(
                    [27.3389, 88.6065]
                ).addTo(map);


                marker.bindPopup(
                    "<b>📍 Selected Location</b><br>" +
                    "Sikkim"
                );


                // ------------------------------------------------
                // UPDATE GRADIO INPUT
                // ------------------------------------------------

                function updateInput(
                    componentId,
                    value
                ) {

                    const component =
                        document.querySelector(
                            "#" + componentId
                        );

                    if (!component) return;

                    const input =
                        component.querySelector("input");

                    if (!input) return;

                    const setter =
                        Object.getOwnPropertyDescriptor(
                            HTMLInputElement.prototype,
                            "value"
                        ).set;

                    setter.call(
                        input,
                        String(value)
                    );

                    input.dispatchEvent(
                        new Event(
                            "input",
                            {bubbles: true}
                        )
                    );

                    input.dispatchEvent(
                        new Event(
                            "change",
                            {bubbles: true}
                        )
                    );
                }


                // ------------------------------------------------
                // MAP CLICK
                // ------------------------------------------------

                map.on(
                    "click",
                    function(event) {

                        const lat =
                            event.latlng.lat.toFixed(6);

                        const lon =
                            event.latlng.lng.toFixed(6);


                        marker.setLatLng(
                            [lat, lon]
                        );


                        updateInput(
                            "latitude_input",
                            lat
                        );


                        updateInput(
                            "longitude_input",
                            lon
                        );


                        marker.bindPopup(
                            "<b>📍 Selected Location</b><br><br>" +
                            "Latitude: " +
                            lat +
                            "<br>" +
                            "Longitude: " +
                            lon
                        ).openPopup();

                    }
                );

             // ------------------------------------------------
                 // SYNC MAP WITH GRADIO COORDINATES
            // ------------------------------------------------

let lastLat = null;
let lastLon = null;

function syncMapWithInputs() {

    const latInput =
        document.querySelector(
            "#latitude_input input"
        );

    const lonInput =
        document.querySelector(
            "#longitude_input input"
        );

    if (!latInput || !lonInput) {
        return;
    }

    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);

    if (
        Number.isNaN(lat) ||
        Number.isNaN(lon)
    ) {
        return;
    }

    // Only move map when coordinates actually change
    if (
        lastLat === lat &&
        lastLon === lon
    ) {
        return;
    }

    lastLat = lat;
    lastLon = lon;

    // Move marker
    marker.setLatLng([
        lat,
        lon
    ]);

    // Move map to selected location
    map.flyTo(
        [lat, lon],
        9,
        {
            duration: 1.2
        }
    );

    // Update popup
    marker.bindPopup(
        "<b>📍 Selected Location</b><br><br>" +
        "Latitude: " +
        lat.toFixed(6) +
        "<br>" +
        "Longitude: " +
        lon.toFixed(6)
    );

}

// ==========================================================
// AUTOMATIC DAY / NIGHT THEME
// ==========================================================
// Check coordinates regularly
setInterval(
    syncMapWithInputs,
    300
);


// Run once after page loads
setTimeout(
    syncMapWithInputs,
    1000
);   

                // ------------------------------------------------
                // MAP SIZE FIX
                // ------------------------------------------------

                setTimeout(
                    function() {
                        map.invalidateSize();
                    },
                    1000
                );

                """
            )


        # ========================================================
    # 🚨 RISK ASSESSMENT
    # ========================================================

    gr.Markdown(
        "## 🚨 Landslide Risk Assessment",
        elem_classes="section-title"
    )

    with gr.Row():

        with gr.Column(elem_classes="risk-panel"):
            risk_level = gr.Textbox(
                label="🚨 Risk Level",
                elem_classes="risk-value",
                interactive=False
            )

        with gr.Column(elem_classes="risk-panel"):
            risk_score = gr.Textbox(
                label="🤖 AI Risk Score",
                elem_classes="risk-value",
                interactive=False
            )

    warning = gr.Textbox(
        label="⚠️ System Warning",
        lines=2,
        interactive=False,
        elem_classes="warning-panel"
    )


    # ========================================================
    # 🌧️ ENVIRONMENTAL CONDITIONS
    # ========================================================

    gr.Markdown(
        "## 🌧️ Environmental Conditions",
        elem_classes="section-title"
    )

    with gr.Row():

        with gr.Column(elem_classes="metric-card"):
            rainfall_3day = gr.Number(
                label="🌧️ 3-Day Rainfall (mm)",
                interactive=False
            )

        with gr.Column(elem_classes="metric-card"):
            rainfall_7day = gr.Number(
                label="🌧️ 7-Day Rainfall (mm)",
                interactive=False
            )

        with gr.Column(elem_classes="metric-card"):
            soil_moisture = gr.Number(
                label="💧 Soil Moisture (m³/m³)",
                interactive=False
            )


    # ========================================================
    # ⛰️ TERRAIN CONDITIONS
    # ========================================================

    gr.Markdown(
        "## ⛰️ Terrain Conditions",
        elem_classes="section-title"
    )

    with gr.Row():

        with gr.Column(elem_classes="metric-card"):
            slope = gr.Number(
                label="📐 Slope (degrees)",
                interactive=False
            )

        with gr.Column(elem_classes="metric-card"):
            elevation = gr.Number(
                label="⛰️ Elevation (m)",
                interactive=False
            )


    # ========================================================
    # 🌡️ CURRENT WEATHER
    # ========================================================

    gr.Markdown(
        "## 🌡️ Current Weather",
        elem_classes="section-title"
    )

    with gr.Row():

        with gr.Column(elem_classes="metric-card"):
            temperature = gr.Number(
                label="🌡️ Temperature (°C)",
                interactive=False
            )

        with gr.Column(elem_classes="metric-card"):
            humidity = gr.Number(
                label="💧 Humidity (%)",
                interactive=False
            )
                # ========================================================
    # 🏆 SIH FOOTER
    # ========================================================

    gr.HTML("""
<div class="dashboard-footer">

    <div>
        <strong style="font-size:15px;">
            🏔️ SIH26001
        </strong>

        <div style="margin-top:5px;">
            Landslide Early Warning System
        </div>

        <div style="margin-top:4px; opacity:0.75;">
            AI-Based Localized Risk Assessment
        </div>
    </div>

    <div>
        <strong>Powered By</strong>

        <div style="margin-top:5px;">
            🌧️ Open-Meteo &nbsp; • &nbsp;
            🗺️ OpenStreetMap &nbsp; • &nbsp;
            🛰️ NASA GLC
        </div>
    </div>

    <div>
        <strong>AI / ML</strong>

        <div style="margin-top:5px;">
            🤖 Gradient Boosting
        </div>
    </div>

</div>
""")
    # ========================================================
    # 🔗 STATE → COORDINATES
    # ========================================================

    state_selector.change(
        fn=select_state,
        inputs=state_selector,
        outputs=[
            latitude,
            longitude
        ]
    )


    # ========================================================
    # 🤖 AI PREDICTION
    # ========================================================

    predict_button.click(
        fn=gradio_prediction,

        inputs=[
            latitude,
            longitude
        ],

        outputs=[
            risk_level,
            risk_score,
            rainfall_3day,
            rainfall_7day,
            soil_moisture,
            slope,
            elevation,
            temperature,
            humidity,
            warning
        ]
    )


# ============================================================
# 🌐 SERVER
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 10000)
    )
)
            

                    
