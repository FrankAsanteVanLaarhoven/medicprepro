import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ModelPerformanceReport:
    def __init__(self, results, df):
        self.results = results
        self.df = df
        
    def generate_performance_dashboard(self):
        """Generate comprehensive model performance report"""
        figs = []
        
        # Model Comparison
        fig1 = self._create_model_comparison()
        figs.append(fig1)
        
        # Feature Importance
        fig2 = self._create_feature_importance()
        figs.append(fig2)
        
        # Prediction Analysis
        fig3 = self._create_prediction_analysis()
        figs.append(fig3)
        
        return figs
        
    def _create_model_comparison(self):
        """Create model performance comparison visualization"""
        fig = go.Figure(data=[
            go.Bar(
                name='R² Score',
                x=list(self.results.keys()),
                y=[r['R2'] for r in self.results.values()],
                text=[f"{r['R2']:.3f}" for r in self.results.values()],
                textposition='auto'
            ),
            go.Bar(
                name='MSE',
                x=list(self.results.keys()),
                y=[r['MSE'] for r in self.results.values()],
                text=[f"{r['MSE']:.3f}" for r in self.results.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            template='plotly_white'
        )
        return fig
