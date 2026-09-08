
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

/* ------------------------------
   MAIN BACKGROUND
--------------------------------*/

body {
    transition: background 0.8s ease;
}

.gradio-container {
    max-width: 1450px !important;
}


/* ------------------------------
   HEADER
--------------------------------*/

.dashboard-header {
    padding: 28px;
    border-radius: 22px;
    margin-bottom: 20px;
    background:
        linear-gradient(
            135deg,
            rgba(0, 137, 123, 0.95),
            rgba(0, 105, 92, 0.90)
        );
    color: white;
    box-shadow: 0 12px 35px rgba(0,0,0,0.18);
}

.dashboard-header h1 {
    font-size: 34px;
    margin: 0;
}

.dashboard-header p {
    font-size: 16px;
    margin-top: 8px;
}


/* ------------------------------
   SECTION HEADINGS
--------------------------------*/

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 10px;
}


/* ------------------------------
   GLASS CARDS
--------------------------------*/

.glass-card {
    border-radius: 18px;
    padding: 18px;
    border: 1px solid rgba(0, 137, 123, 0.30);
    box-shadow: 0 8px 25px rgba(0,0,0,0.10);
}


/* ------------------------------
   RISK CARD
--------------------------------*/

.risk-card {
    border-radius: 20px;
    padding: 22px;
    border: 2px solid rgba(0, 137, 123, 0.35);
    text-align: center;
}


/* ------------------------------
   MAP
--------------------------------*/

#landslide-map {
    height: 500px !important;
    width: 100% !important;
    border-radius: 20px;
    overflow: hidden;
    border: 2px solid rgba(0, 137, 123, 0.45);
    box-shadow: 0 12px 30px rgba(0,0,0,0.20);
}


/* ------------------------------
   BUTTON
--------------------------------*/

.analyze-button {
    border-radius: 12px !important;
    font-weight: 700 !important;
}


/* ------------------------------
   STATUS
--------------------------------*/

.status-pill {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 20px;
    font-weight: 700;
    background: rgba(0, 137, 123, 0.12);
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

        <h1>
            🚨 Landslide Early Warning System
        </h1>

        <p>
            AI-Based Localized Landslide Risk Assessment
            for Northeast India
        </p>

        <div style="
            margin-top:12px;
            font-size:14px;
            opacity:0.95;
        ">
            🌧️ Rainfall &nbsp; • &nbsp;
            💧 Soil Moisture &nbsp; • &nbsp;
            ⛰️ Terrain &nbsp; • &nbsp;
            🤖 AI Risk Assessment
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

        with gr.Column():

            risk_level = gr.Textbox(
                label="Risk Level",
                elem_classes="risk-card"
            )

        with gr.Column():

            risk_score = gr.Textbox(
                label="AI Risk Score",
                elem_classes="risk-card"
            )


    warning = gr.Textbox(
        label="⚠️ System Warning",
        lines=2
    )


    # ========================================================
    # 🌧️ ENVIRONMENTAL CONDITIONS
    # ========================================================

    gr.Markdown(
        "## 🌧️ Environmental Conditions",
        elem_classes="section-title"
    )

    with gr.Row():

        rainfall_3day = gr.Number(
            label="3-Day Rainfall (mm)"
        )

        rainfall_7day = gr.Number(
            label="7-Day Rainfall (mm)"
        )

        soil_moisture = gr.Number(
            label="Soil Moisture (m³/m³)"
        )


    # ========================================================
    # ⛰️ TERRAIN
    # ========================================================

    gr.Markdown(
        "## ⛰️ Terrain Conditions",
        elem_classes="section-title"
    )

    with gr.Row():

        slope = gr.Number(
            label="Slope (degrees)"
        )

        elevation = gr.Number(
            label="Elevation (m)"
        )


    # ========================================================
    # 🌡️ WEATHER
    # ========================================================

    gr.Markdown(
        "## 🌡️ Current Weather",
        elem_classes="section-title"
    )

    with gr.Row():

        temperature = gr.Number(
            label="Temperature (°C)"
        )

        humidity = gr.Number(
            label="Humidity (%)"
        )


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
            

                    
