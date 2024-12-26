import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.express as px
import logging

class ModelEvaluationService:
    def __init__(self, models_dict):
        self.models = models_dict
        self.metrics_history = {}
        self.logger = logging.getLogger(__name__)
        
    def evaluate_all_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        evaluation_results = {}
        
        for name, model in self.models.items():
            try:
                predictions = model.predict(X_test)
                importance = self._get_feature_importance(model, X_test.columns)
                
                evaluation_results[name] = {
                    'MSE': mean_squared_error(y_test, predictions),
                    'MAE': mean_absolute_error(y_test, predictions),
                    'R2': r2_score(y_test, predictions),
                    'Feature_Importance': importance,
                    'Predictions': predictions
                }
                
                self.metrics_history[name] = self._track_metrics(
                    evaluation_results[name]
                )
                
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {str(e)}")
                
        return evaluation_results

    def _get_feature_importance(self, model, feature_names):
        """Extract feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            return pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def _track_metrics(self, metrics):
        """Track metrics over time"""
        return {
            'timestamp': datetime.now(),
            'metrics': metrics
        }
