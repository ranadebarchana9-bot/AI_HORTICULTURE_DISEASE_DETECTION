import streamlit as st
import numpy as np
from PIL import Image
import requests
import time

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

from google import genai

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Horticulture Crop Disease Advisory System",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 AI-Based Horticulture Crop Disease Detection & Advisory System")
st.caption("Deep Learning + Gemini AI + Live Weather Context")

# ---------------------------------------------------------
# SIDEBAR: API KEYS (entered by user, not hardcoded)
# ---------------------------------------------------------
st.sidebar.header("🔑 API Keys")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
weather_api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

st.sidebar.markdown("---")
st.sidebar.info(
    "Your keys are used only for this session and are never stored or sent anywhere else."
)

# ---------------------------------------------------------
# LOAD MODEL AND CLASS NAMES (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("plant_disease_model_v2_extended.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_model_and_classes()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        "Could not load the model files. Make sure "
        "'plant_disease_model_v2_extended.keras' and 'class_names.json' "
        "are in the same folder as this app.py file."
    )
    st.exception(e)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def predict_disease(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


def get_weather(city_name, api_key):
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
            }
        return None
    except Exception:
        return None


def get_advisory(disease_name, weather_context, api_key, retries=3, delay=4):
    client = genai.Client(api_key=api_key, vertexai=False)

    prompt = f"""You are an agricultural expert. A plant has been diagnosed with: {disease_name.replace('___', ' - ').replace('_', ' ')}

{weather_context}

Provide a concise advisory in this format:
1. **Cause**: Brief explanation
2. **Treatment**: Practical treatment steps
3. **Prevention**: How to prevent this in future, considering the current weather conditions if relevant

Keep it practical and farmer-friendly, under 150 words total."""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
            )
            return response.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return f"Could not generate advisory right now. Please try again. ({e})"


# ---------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"]
    )

with col2:
    city_name = st.text_input("Enter your city (for weather context)", value="")

analyze_clicked = st.button("🔍 Analyze Plant", type="primary", use_container_width=True)

if analyze_clicked:
    if not model_loaded:
        st.stop()
    if uploaded_file is None:
        st.warning("Please upload a leaf image first.")
        st.stop()
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
        st.stop()

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        disease, confidence = predict_disease(img)

    disease_display = disease.replace("___", " — ").replace("_", " ")

    st.subheader("🩺 Diagnosis")
    st.write(f"**Predicted Disease:** {disease_display}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    weather = None
    weather_context = "Weather data unavailable."
    if city_name and weather_api_key:
        with st.spinner("Fetching current weather..."):
            weather = get_weather(city_name, weather_api_key)
        if weather:
            weather_context = (
                f"Current weather in {city_name}: {weather['temperature']}°C, "
                f"{weather['humidity']}% humidity, {weather['description']}."
            )

    if weather:
        st.subheader("🌤️ Current Weather")
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperature", f"{weather['temperature']} °C")
        c2.metric("Humidity", f"{weather['humidity']} %")
        c3.metric("Condition", weather["description"].title())
    elif city_name and weather_api_key:
        st.info("Weather data unavailable for this city — advisory will proceed without it.")

    with st.spinner("Generating advisory..."):
        advisory = get_advisory(disease, weather_context, gemini_api_key)

    st.subheader("📋 Advisory")
    st.markdown(advisory)

st.markdown("---")
st.caption(
    "AI-Based Horticulture Crop Disease Detection and Advisory System — "
    "Deep Learning + Gemini AI + Basic Environmental Context"
)
