# src/app.py
import streamlit as st
import plotly.express as px
import pandas as pd
from pathlib import Path
import logging
from utils.data_loader import DataLoader
from components.analysis import MedicineAnalyzer
from components.monitoring import ModelMonitor
from components.predictor import MedicinePredictionService

class MedicProDashboard:
    def __init__(self):
        self.setup_logging()
        self.setup_app()
        self.initialize_components()

    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            filename='logs/app.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_app(self):
        """Configure Streamlit app settings"""
        st.set_page_config(
            page_title="MedicPrepro",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self.load_css()
        self.initialize_session_state()

    def initialize_components(self):
        """Initialize app components"""
        self.data_loader = DataLoader()
        self.monitor = ModelMonitor()
        self.predictor = MedicinePredictionService('models/random_forest.joblib')

    def initialize_session_state(self):
        """Initialize session state with error handling"""
        try:
            if 'data' not in st.session_state:
                st.session_state.data = None
            if 'data_source' not in st.session_state:
                st.session_state.data_source = None
            if 'analyzer' not in st.session_state:
                st.session_state.analyzer = None
            if 'model_loaded' not in st.session_state:
                st.session_state.model_loaded = False
        except Exception as e:
            self.logger.error(f"Session state initialization failed: {str(e)}")
            raise

    def load_data_section(self):
        """Enhanced data loading section with progress tracking"""
        st.sidebar.header("Data Source")
        
        # Add file size validation
        MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

        data_source = st.sidebar.radio(
            "Choose data source",
            ["Upload CSV", "Kaggle Dataset", "Sample Data"]
        )

        if data_source == "Upload CSV":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="Upload your medicine dataset (max 200MB)"
            )
            if uploaded_file:
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.error("File size exceeds 200MB limit.")
                    return
                self._handle_file_upload(uploaded_file)

        elif data_source == "Kaggle Dataset":
            if st.sidebar.button("Load Kaggle Dataset"):
                self._handle_kaggle_download()

    def render_predictions(self):
        """Enhanced prediction interface"""
        if not st.session_state.data:
            st.warning("Please load data first.")
            return
            
        st.title("Medicine Effectiveness Prediction")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                composition_count = st.number_input(
                    "Number of Ingredients",
                    min_value=1,
                    max_value=10,
                    help="Enter number of active ingredients (1-10)"
                )
                side_effects = st.number_input(
                    "Number of Side Effects",
                    min_value=0,
                    max_value=20,
                    help="Enter number of known side effects"
                )
            
            with col2:
                satisfaction = st.slider(
                    "Patient Satisfaction Score",
                    0, 100, 50,
                    help="Enter average patient satisfaction score"
                )
                manufacturer_rating = st.slider(
                    "Manufacturer Rating",
                    0, 100, 50,
                    help="Enter manufacturer's quality rating"
                )
            
            submitted = st.form_submit_button("Predict Effectiveness")
            
            if submitted:
                try:
                    features = {
                        'composition_count': composition_count,
                        'side_effects': side_effects,
                        'satisfaction': satisfaction,
                        'manufacturer_rating': manufacturer_rating
                    }
                    
                    prediction = self.predictor.predict(features)
                    
                    st.success(f"Predicted Effectiveness: {prediction['prediction']:.1f}%")
                    st.info(f"Confidence Score: {prediction['confidence']:.2f}")
                    
                    # Log prediction
                    self.monitor.track_prediction(features, prediction['prediction'])
                    
                except Exception as e:
                    self.logger.error(f"Prediction failed: {str(e)}")
                    st.error("Prediction failed. Please try again.")

    def run(self):
        """Main application loop with error handling"""
        try:
            self.load_data_section()
            
            if st.session_state.data is not None:
                page = st.sidebar.selectbox(
                    "Navigation",
                    ["Overview", "EDA Report", "Model Analysis", "Predictions"]
                )
                
                if page == "Overview":
                    self.render_overview()
                elif page == "EDA Report":
                    self.render_eda_report()
                elif page == "Model Analysis":
                    self.render_model_analysis()
                elif page == "Predictions":
                    self.render_predictions()
                    
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    app = MedicProDashboard()
    app.run()
