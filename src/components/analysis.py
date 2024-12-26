import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

class MedicineAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.reports_dir = Path('reports')
        self.reports_dir.mkdir(exist_ok=True)

    def generate_profile_report(self):
        """Generate comprehensive EDA report"""
        profile = ProfileReport(
            self.df,
            title="Medicine Analysis Report",
            explorative=True,
            correlations={
                "auto": {"calculate": True},
                "pearson": {"calculate": True},
                "spearman": {"calculate": True}
            }
        )
        
        report_path = self.reports_dir / 'medicine_analysis.html'
        profile.to_file(report_path)
        return profile.to_html()

    def create_analysis_dashboard(self):
        """Create interactive analysis dashboard"""
        figures = []
        
        # Review Distribution
        fig1 = self._create_review_distribution()
        figures.append(fig1)
        
        # Manufacturer Performance
        fig2 = self._create_manufacturer_analysis()
        figures.append(fig2)
        
        # Side Effects Analysis
        fig3 = self._create_side_effects_analysis()
        figures.append(fig3)
        
        return figures

    def _create_review_distribution(self):
        """Create review distribution visualization"""
        fig = go.Figure()
        
        for review_type in ['excellent_review_%', 'average_review_%', 'poor_review_%']:
            fig.add_trace(
                go.Histogram(
                    x=self.df[review_type],
                    name=review_type.replace('_', ' ').title(),
                    nbinsx=50
                )
            )
            
        fig.update_layout(
            title='Distribution of Reviews',
            barmode='overlay',
            xaxis_title='Review Percentage',
            yaxis_title='Count'
        )
        
        return fig

    def _create_manufacturer_analysis(self):
        """Create manufacturer performance analysis"""
        top_manufacturers = (
            self.df.groupby('manufacturer')['excellent_review_%']
            .agg(['mean', 'count'])
            .sort_values('mean', ascending=False)
            .head(10)
        )
        
        fig = px.bar(
            top_manufacturers,
            y=top_manufacturers.index,
            x='mean',
            title='Top 10 Manufacturers by Average Review Score',
            labels={'mean': 'Average Excellent Review %'}
        )
        
        return fig

    def _create_side_effects_analysis(self):
        """Create side effects analysis visualization"""
        self.df['side_effects_count'] = self.df['side_effects'].str.count(',') + 1
        
        fig = px.scatter(
            self.df,
            x='side_effects_count',
            y='excellent_review_%',
            color='manufacturer',
            title='Side Effects vs Review Score',
            labels={
                'side_effects_count': 'Number of Side Effects',
                'excellent_review_%': 'Excellent Review %'
            }
        )
        
        return fig
