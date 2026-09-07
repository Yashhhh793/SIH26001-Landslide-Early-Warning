
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
# 🎨 GRADIO FUNCTION
# =======================================================

def gradio_prediction(lat, lon):

    try:

        result = predict_live_landslide_risk(
            float(lat),
            float(lon)
        )

        risk_level = result["risk_level"]
        risk_score = result["ai_risk_score"]

        if risk_level == "HIGH RISK":

            warning = (
                "🚨 High landslide risk detected. "
                "Exercise caution and monitor local warnings."
            )

        elif risk_level == "MODERATE RISK":

            warning = (
                "⚠️ Moderate landslide risk. "
                "Stay alert and monitor weather conditions."
            )

        else:

            warning = (
                "✅ Lower landslide risk under "
                "current conditions."
            )

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
# 🗺️ FINAL GRADIO UI WITH INTERACTIVE MAP
# ============================================================

with gr.Blocks(
    title="Landslide Early Warning System"
) as demo:

    gr.Markdown("""
# 🚨 Landslide Early Warning System

### AI-Based Localized Landslide Risk Assessment

Select a location on the map or enter latitude and longitude manually.
The system fetches live environmental and terrain data and calculates
the localized landslide risk.
""")

    # ========================================================
    # LOCATION
    # ========================================================

    gr.Markdown("## 📍 Select Location")

    with gr.Row():

        # ----------------------------------------------------
        # LATITUDE / LONGITUDE
        # ----------------------------------------------------

        with gr.Column(scale=1):

            latitude = gr.Number(
                label="Latitude",
                value=27.3389,
                elem_id="latitude_input"
            )

            longitude = gr.Number(
                label="Longitude",
                value=88.6065,
                elem_id="longitude_input"
            )

            gr.Markdown(
                "💡 Click anywhere on the map to select a location."
            )

            predict_button = gr.Button(
                "🔍 Analyze Landslide Risk",
                variant="primary"
            )

        # ----------------------------------------------------
        # INTERACTIVE MAP
        # ----------------------------------------------------

        with gr.Column(scale=2):

            map_html = gr.HTML(
                html_template="""
                <div id="landslide-map"
                     style="
                        width:100%;
                        height:450px;
                        border-radius:12px;
                        overflow:hidden;
                        border:2px solid #777;
                     ">
                </div>
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

                // Create map
                const map = L.map(mapContainer).setView(
                    [27.3389, 88.6065],
                    7
                );

                // OpenStreetMap
                L.tileLayer(
                    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    {
                        attribution:
                            "&copy; OpenStreetMap contributors"
                    }
                ).addTo(map);


                // Initial marker
                let marker = L.marker(
                    [27.3389, 88.6065]
                ).addTo(map);

                marker.bindPopup(
                    "<b>Selected Location</b><br>" +
                    "Latitude: 27.3389<br>" +
                    "Longitude: 88.6065"
                );


                // ------------------------------------------------
                // UPDATE GRADIO NUMBER INPUT
                // ------------------------------------------------

                function updateGradioInput(
                    componentId,
                    value
                ) {

                    const component =
                        document.querySelector(
                            "#" + componentId
                        );

                    if (!component) {
                        console.log(
                            "Component not found:",
                            componentId
                        );
                        return;
                    }

                    const input =
                        component.querySelector("input");

                    if (!input) {
                        console.log(
                            "Input not found:",
                            componentId
                        );
                        return;
                    }


                    const nativeSetter =
                        Object.getOwnPropertyDescriptor(
                            HTMLInputElement.prototype,
                            "value"
                        ).set;


                    nativeSetter.call(
                        input,
                        String(value)
                    );


                    input.dispatchEvent(
                        new Event(
                            "input",
                            {
                                bubbles: true
                            }
                        )
                    );


                    input.dispatchEvent(
                        new Event(
                            "change",
                            {
                                bubbles: true
                            }
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


                        // Move marker
                        marker.setLatLng(
                            [lat, lon]
                        );


                        // Update latitude
                        updateGradioInput(
                            "latitude_input",
                            lat
                        );


                        // Update longitude
                        updateGradioInput(
                            "longitude_input",
                            lon
                        );


                        // Popup
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


                // Fix map size after rendering
                setTimeout(
                    function() {
                        map.invalidateSize();
                    },
                    1000
                );

                """
            )


    # ========================================================
    # LANDSLIDE RISK ASSESSMENT
    # ========================================================

    gr.Markdown("## 🚨 Landslide Risk Assessment")

    with gr.Row():

        risk_level = gr.Textbox(
            label="Risk Level"
        )

        risk_score = gr.Textbox(
            label="Landslide Risk Score"
        )


    warning = gr.Textbox(
        label="System Warning",
        lines=2
    )


    # ========================================================
    # ENVIRONMENTAL CONDITIONS
    # ========================================================

    gr.Markdown("## 🌧️ Environmental Conditions")

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
    # TERRAIN CONDITIONS
    # ========================================================

    gr.Markdown("## ⛰️ Terrain Conditions")

    with gr.Row():

        slope = gr.Number(
            label="Slope (degrees)"
        )

        elevation = gr.Number(
            label="Elevation (m)"
        )


    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    gr.Markdown("## 🌡️ Current Weather")

    with gr.Row():

        temperature = gr.Number(
            label="Temperature (°C)"
        )

        humidity = gr.Number(
            label="Humidity (%)"
        )


    # ========================================================
    # EXISTING PREDICTION FUNCTION
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
# 🌐 SERVER LAUNCH
# ============================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=int(
        os.environ.get("PORT", 10000)
    )
)

    
