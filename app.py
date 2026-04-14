import streamlit as st
import pickle
import numpy as np
# Page configuration
st.set_page_config(
    page_title="AI Crop Advisor",
    layout="wide"
)
# Load trained model
model = pickle.load(open("crop_model.pkl","rb"))
encoder = pickle.load(open("label_encoder.pkl","rb"))
# TITLE SECTION
st.markdown(
"""
<h1 style="font-size:56px; margin-bottom:0;"> AI Crop Advisor</h1>
<p style="font-size:26px; color:gray;">
Smart Crop Recommendation using Machine Learning
</p>
""",
unsafe_allow_html=True
)
st.markdown("---")
col1, col2 = st.columns([1,2])
# INPUT SECTION
with col1:
    st.markdown("<h2 style='font-size:34px;'>Input Soil Parameters</h2>", unsafe_allow_html=True)
    N = st.slider("Nitrogen (N)",0,150,50)
    P = st.slider("Phosphorus (P)",0,150,40)
    K = st.slider("Potassium (K)",0,150,40)
    temperature = st.slider("Temperature (°C)",0,50,25)
    humidity = st.slider("Humidity (%)",0,100,60)
    ph = st.slider("Soil pH",0.0,14.0,6.5)
    rainfall = st.slider("Rainfall (mm)",0,500,200)
    predict = st.button(" Predict Crop")
# RESULT SECTION
with col2:
    st.markdown("<h2 style='font-size:34px;'>Prediction Result</h2>", unsafe_allow_html=True)
    if predict:
        data = np.array([[N,P,K,temperature,humidity,ph,rainfall]])
        prediction = model.predict(data)
        crop = encoder.inverse_transform(prediction)
        # Prediction card
        st.markdown(
        f"""
        <div style="background-color:#e8f5e9;padding:25px;border-radius:12px;">
        <h2 style="color:#2e7d32;font-size:40px;">
         Recommended Crop: {crop[0].upper()}
        </h2>
        </div>
        """,
        unsafe_allow_html=True
        )
        st.markdown(
        f"""
        <p style="font-size:26px;">
        Based on the given soil nutrients and environmental conditions, the machine learning model predicts that 
        <b>{crop[0]}</b> is the most suitable crop for cultivation.
        </p>
        """,
        unsafe_allow_html=True
        )
    st.markdown("---")
    # System Information
    st.markdown(
    """
    <h3 style="font-size:34px;">System Information</h3>
    <p style="font-size:24px;">
    This system uses machine learning to recommend suitable crops based on soil nutrients and environmental conditions.
    The prediction is made using important agricultural factors such as nitrogen, phosphorus, potassium, temperature,
    humidity, soil pH, and rainfall.
    </p>
    """,
    unsafe_allow_html=True
    )