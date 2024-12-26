import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
from ydata_profiling import ProfileReport
import plotly.express as px

class MedicineModelManager:
    def __init__(self, model_path='models/random_forest.joblib'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        
    def load_model(self):
        """Load trained model with error handling"""
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None
            
    def preprocess_input(self, composition_count, side_effects, satisfaction, manufacturer_rating):
        """Preprocess input data with validation"""
        try:
            features = np.array([[
                composition_count,
                side_effects,
                satisfaction,
                manufacturer_rating
            ]])
            return self.scaler.fit_transform(features)
        except Exception as e:
            st.error(f"Preprocessing failed: {str(e)}")
            return None
            
    def save_model(self, model):
        """Save model with version control"""
        try:
            joblib.dump(model, self.model_path)
            st.success("Model saved successfully")
        except Exception as e:
            st.error(f"Failed to save model: {str(e)}")
