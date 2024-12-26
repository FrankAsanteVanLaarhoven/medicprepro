import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FeatureAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def create_feature_importance_plot(self, model):
        """Visualize feature importance"""
        if not hasattr(model, 'feature_importances_'):
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.df.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            importance_df,
            x='feature',
            y='importance',
            title='Feature Importance Analysis'
        )
        return fig
        
    def create_correlation_heatmap(self):
        """Create correlation analysis visualization"""
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Heatmap',
            color_continuous_scale='RdBu'
        )
        return fig
