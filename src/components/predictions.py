import streamlit as st
import numpy as np
from utils.model_utils import load_model, preprocess_input

def render_predictions():
    st.title("Medicine Effectiveness Predictor")
    
    with st.form("prediction_form"):
        # Input fields
        composition_count = st.number_input(
            "Number of Ingredients",
            min_value=1,
            max_value=10
        )
        
        side_effects = st.number_input(
            "Number of Side Effects",
            min_value=0
        )
        
        satisfaction = st.slider(
            "Patient Satisfaction Score",
            0, 100, 50
        )
        
        manufacturer_rating = st.slider(
            "Manufacturer Rating",
            0, 100, 50
        )
        
        submitted = st.form_submit_button("Predict Effectiveness")
        
        if submitted:
            try:
                model = load_model()
                input_data = preprocess_input(
                    composition_count,
                    side_effects,
                    satisfaction,
                    manufacturer_rating
                )
                prediction = model.predict(input_data)[0]
                st.success(f"Predicted Effectiveness: {prediction:.1f}%")
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
