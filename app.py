"""
app.py - Streamlit dashboard for Smart Irrigation Prediction.
Run with:  streamlit run app.py
"""
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sensor_simulator import get_sensor_reading
from train_model import FEATURE_COLUMNS, MODEL_FILE, main as train_model

st.set_page_config(page_title="Smart Irrigation", page_icon="🌱")


@st.cache_resource
def load_model():
    """Load the saved model; train it first if the file is missing."""
    if not os.path.exists(MODEL_FILE):
        train_model()
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)


bundle = load_model()
model = bundle["model"]

# ---------- Session state (keeps values between button clicks) ----------
st.session_state.setdefault("moisture", 40)
st.session_state.setdefault("temperature", 28)
st.session_state.setdefault("humidity", 60)
st.session_state.setdefault("history", [])


def use_simulated_reading():
    """Button callback: put a new simulated sensor reading into the inputs."""
    reading = get_sensor_reading()
    st.session_state.moisture = reading["soil_moisture"]
    st.session_state.temperature = reading["temperature"]
    st.session_state.humidity = reading["humidity"]


# ---------- Title ----------
st.title("🌱 Smart Irrigation Prediction")
st.write("An IoT / Edge ML demo: a Decision Tree predicts whether a field needs "
         "irrigation from soil moisture, temperature and humidity. "
         "Sensor data is **simulated** (no physical hardware).")

# ---------- Inputs ----------
st.subheader("Sensor Inputs")
st.button("Generate Simulated Sensor Readings", on_click=use_simulated_reading)
st.slider("Soil Moisture (%)", 0, 100, key="moisture")
st.slider("Temperature (°C)", 10, 45, key="temperature")
st.slider("Humidity (%)", 10, 100, key="humidity")

# ---------- Current readings ----------
st.subheader("Current Sensor Readings")
c1, c2, c3 = st.columns(3)
c1.metric("Soil Moisture", f"{st.session_state.moisture} %")
c2.metric("Temperature", f"{st.session_state.temperature} °C")
c3.metric("Humidity", f"{st.session_state.humidity} %")

# ---------- Prediction ----------
if st.button("Predict Irrigation"):
    # Same column names and order as in train_model.py
    row = pd.DataFrame([[st.session_state.moisture,
                         st.session_state.temperature,
                         st.session_state.humidity]], columns=FEATURE_COLUMNS)
    result = int(model.predict(row)[0])
    label = "Irrigation Required" if result == 1 else "Irrigation Not Required"
    if result == 1:
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")
    st.session_state.history.append({
        "Soil Moisture (%)": st.session_state.moisture,
        "Temperature (°C)": st.session_state.temperature,
        "Humidity (%)": st.session_state.humidity,
        "Prediction": label,
    })

# ---------- Model accuracy ----------
st.subheader("Model Accuracy")
st.write(f"Test-set accuracy: **{bundle['accuracy']:.2%}** "
         "(measured on simulated data when `train_model.py` was run).")

# ---------- Recent predictions ----------
st.subheader("Recent Predictions")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history[-10:]).iloc[::-1],
                 hide_index=True)
else:
    st.info("No predictions yet. Click 'Predict Irrigation'.")

# ---------- Simple visualization: what the model relies on ----------
st.subheader("Feature Importance")
fig, ax = plt.subplots(figsize=(5, 2.5))
ax.barh(FEATURE_COLUMNS, model.feature_importances_, color="seagreen")
ax.set_xlabel("Importance")
st.pyplot(fig)
