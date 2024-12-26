import logging
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prediction_history = []
        self.model_metrics = {}
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='logs/model_monitoring.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/model_monitoring.log'),
                logging.StreamHandler()
            ]
        )

    def track_prediction(self, features, prediction, actual=None):
        """Track prediction with performance metrics"""
        record = {
            'timestamp': datetime.now(),
            'features': features,
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual) if actual else None
        }
        self.prediction_history.append(record)
        self._check_model_drift(record)
        
    def _check_model_drift(self, record):
        """Monitor model drift"""
        recent_records = self.prediction_history[-100:]
        if len(recent_records) >= 100:
            recent_errors = [r['error'] for r in recent_records if r['error']]
            if np.mean(recent_errors) > 0.1:
                self.logger.warning("Potential model drift detected")

        
    def generate_monitoring_dashboard(self):
        """Create comprehensive monitoring visualizations"""
        if not self.predictions_history:
            return None
            
        df = pd.DataFrame(self.predictions_history)
        
        figs = []
        # Prediction distribution
        figs.append(px.histogram(df, x='prediction', 
                               title='Prediction Distribution',
                               marginal='box'))
        
        # Confidence trend
        figs.append(px.line(df, x='timestamp', y='confidence',
                           title='Confidence Trend'))
        
        # Error analysis if actuals available
        if df['actual'].notna().any():
            figs.append(px.scatter(df, x='actual', y='prediction',
                                 title='Predicted vs Actual'))
            
        return figs

# Enhanced Feature Analyzer
class EnhancedFeatureAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def analyze_features(self, model=None):
        """Comprehensive feature analysis"""
        figs = []
        
        # Feature importance if model available
        if model and hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.df.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            figs.append(px.bar(importance_df,
                             x='feature', y='importance',
                             title='Feature Importance Analysis'))
            
        # Correlation analysis
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        figs.append(px.imshow(corr_matrix,
                            title='Feature Correlation Analysis',
                            color_continuous_scale='RdBu'))
        
        # Feature distributions
        for col in numeric_cols:
            figs.append(px.histogram(self.df, x=col,
                                   title=f'{col} Distribution',
                                   marginal='box'))
            
        return figs

# Enhanced Medicine Visualizer
class EnhancedMedicineVisualizer:
    def __init__(self, df):
        self.df = df
        
    def create_comprehensive_analysis(self):
        """Generate comprehensive medicine analysis"""
        figs = []
        
        # Review distribution
        fig = make_subplots(rows=1, cols=3,
                           subplot_titles=('Excellent', 'Average', 'Poor'))
        
        review_cols = ['excellent_review_%', 'average_review_%', 'poor_review_%']
        for i, col in enumerate(review_cols, 1):
            fig.add_trace(
                go.Histogram(x=self.df[col],
                           name=col.split('_')[0].title(),
                           nbinsx=50),
                row=1, col=i
            )
        figs.append(fig)
        
        # Manufacturer analysis
        top_manufacturers = (
            self.df.groupby('manufacturer')
            .agg({
                'excellent_review_%': ['mean', 'count', 'std'],
                'side_effects_count': 'mean'
            })
            .sort_values(('excellent_review_%', 'mean'), ascending=False)
            .head(10)
        )
        
        figs.append(px.bar(top_manufacturers,
                          y=top_manufacturers.index,
                          x=('excellent_review_%', 'mean'),
                          error_x=('excellent_review_%', 'std'),
                          title='Top Manufacturers Analysis'))
        
        return figs
