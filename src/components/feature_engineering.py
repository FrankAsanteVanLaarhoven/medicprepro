from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        
    def create_features(self):
        """Create all features"""
        return (self.df
                .pipe(self._create_composition_features)
                .pipe(self._create_review_features)
                .pipe(self._create_manufacturer_features)
                .pipe(self._scale_features))
    
    def _create_composition_features(self, df):
        df['composition_count'] = df['composition'].str.count(',') + 1
        df['composition_complexity'] = df['composition'].str.len()
        return df
        
    def _create_review_features(self, df):
        df['satisfaction_score'] = (
            0.5 * df['excellent_review_%'] +
            0.3 * df['average_review_%'] +
            0.2 * (100 - df['poor_review_%'])
        ) / 100
        return df
