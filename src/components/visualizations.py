# src/components/visualization.py
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Optional

class MedicineAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = self._setup_logger()
        self._validate_dataframe()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure comprehensive logging"""
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler('logs/analysis.log')
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _validate_dataframe(self) -> None:
        """Validate required columns and data types"""
        required_cols = [
            'excellent_review_%', 'average_review_%', 'poor_review_%',
            'manufacturer', 'side_effects_count'
        ]
        if not all(col in self.df.columns for col in required_cols):
            raise ValueError("Missing required columns in DataFrame")
            
        # Validate review percentages
        for col in ['excellent_review_%', 'average_review_%', 'poor_review_%']:
            if not self.df[col].between(0, 100).all():
                self.logger.warning(f"Invalid percentages found in {col}")
                self.df[col] = self.df[col].clip(0, 100)

    def create_analysis_dashboard(self) -> List[go.Figure]:
        """Generate comprehensive analysis dashboard"""
        try:
            figs = []
            
            # Review Distribution
            fig_reviews = self._create_review_distribution()
            figs.append(fig_reviews)
            
            # Manufacturer Analysis 
            fig_mfr = self._create_manufacturer_analysis()
            figs.append(fig_mfr)
            
            # Side Effects Impact
            fig_side_effects = self._create_side_effects_analysis()
            figs.append(fig_side_effects)
            
            # Satisfaction Trends
            fig_trends = self._create_satisfaction_trends()
            figs.append(fig_trends)
            
            # Correlation Analysis
            fig_corr = self._create_correlation_analysis()
            figs.append(fig_corr)
            
            self.logger.info("Analysis dashboard generated successfully")
            return figs
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {str(e)}")
            raise

    def _create_review_distribution(self) -> go.Figure:
        """Create enhanced review distribution visualization"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Excellent', 'Average', 'Poor'),
            specs=[[{'type': 'histogram'}] * 3]
        )
        
        review_cols = ['excellent_review_%', 'average_review_%', 'poor_review_%']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for i, (col, color) in enumerate(zip(review_cols, colors), 1):
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col.split('_')[0].title(),
                    nbinsx=50,
                    marker_color=color,
                    customdata=np.stack((
                        self.df['manufacturer'],
                        self.df['side_effects_count']
                    ), axis=-1),
                    hovertemplate=(
                        "<b>%{x:.1f}%</b> Reviews<br>"
                        "Manufacturer: %{customdata[0]}<br>"
                        "Side Effects: %{customdata[1]}<extra></extra>"
                    )
                ),
                row=1, col=i
            )
        
        fig.update_layout(
            height=500,
            title_text="Review Distribution Analysis",
            showlegend=False,
            bargap=0.1,
            template="plotly_white"
        )
        
        return fig

    def _create_manufacturer_analysis(self) -> go.Figure:
        """Create enhanced manufacturer performance analysis"""
        top_manufacturers = (
            self.df.groupby('manufacturer')
            .agg({
                'excellent_review_%': ['mean', 'count', 'std'],
                'side_effects_count': ['mean', 'std'],
                'satisfaction_score': 'mean'
            })
            .sort_values(('excellent_review_%', 'mean'), ascending=False)
            .head(10)
        )
        
        fig = px.bar(
            top_manufacturers,
            y=top_manufacturers.index,
            x=('excellent_review_%', 'mean'),
            error_x=('excellent_review_%', 'std'),
            title='Top Manufacturers Performance',
            labels={
                'y': 'Manufacturer',
                'x': 'Average Excellent Review %'
            },
            color=('satisfaction_score', 'mean'),
            color_continuous_scale='RdYlBu',
            hover_data={
                ('side_effects_count', 'mean'): ':.2f',
                ('excellent_review_%', 'count'): True
            }
        )
        
        fig.update_layout(template="plotly_white")
        return fig
