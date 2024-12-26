import streamlit as st
import plotly.express as px

class PerformanceDashboard:
    def __init__(self, results):
        self.results = results
        
    def render_metrics(self):
        """Display key performance metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Model R²",
                f"{max(r['R2'] for r in self.results.values()):.3f}"
            )
        with col2:
            st.metric(
                "Lowest MSE",
                f"{min(r['MSE'] for r in self.results.values()):.3f}"
            )
        with col3:
            st.metric(
                "Average MAE",
                f"{sum(r['MAE'] for r in self.results.values())/len(self.results):.3f}"
            )
            
    def render_comparison_plot(self):
        """Create model comparison visualization"""
        fig = go.Figure(data=[
            go.Bar(
                name='R² Score',
                x=list(self.results.keys()),
                y=[r['R2'] for r in self.results.values()]
            ),
            go.Bar(
                name='MSE',
                x=list(self.results.keys()),
                y=[r['MSE'] for r in self.results.values()]
            )
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group'
        )
        return fig
