class DataQualityReport:
    def __init__(self, df):
        self.df = df
        
    def generate_quality_report(self):
        """Generate comprehensive data quality report"""
        report = {
            'completeness': self._check_completeness(),
            'validity': self._check_validity(),
            'consistency': self._check_consistency(),
            'distribution': self._analyze_distributions()
        }
        return report
        
    def _check_completeness(self):
        """Check data completeness"""
        return {
            'missing_values': self.df.isnull().sum().to_dict(),
            'completion_rate': (1 - self.df.isnull().mean()).to_dict()
        }
        
    def _check_validity(self):
        """Check data validity"""
        validity = {}
        review_cols = ['excellent_review_%', 'average_review_%', 'poor_review_%']
        
        for col in review_cols:
            validity[col] = {
                'in_range': (self.df[col].between(0, 100)).mean(),
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }
        return validity
