class EffectivenessReport:
    def __init__(self, df):
        self.df = df
        
    def generate_effectiveness_report(self):
        """Generate medicine effectiveness analysis report"""
        figs = []
        
        # Review Distribution
        fig1 = self._create_review_distribution()
        figs.append(fig1)
        
        # Manufacturer Performance
        fig2 = self._create_manufacturer_analysis()
        figs.append(fig2)
        
        # Side Effects Impact
        fig3 = self._create_side_effects_analysis()
        figs.append(fig3)
        
        return figs
        
    def _create_review_distribution(self):
        """Create review distribution visualization"""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Excellent', 'Average', 'Poor')
        )
        
        review_cols = ['excellent_review_%', 'average_review_%', 'poor_review_%']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        for i, (col, color) in enumerate(zip(review_cols, colors), 1):
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col.split('_')[0].title(),
                    marker_color=color
                ),
                row=1, col=i
            )
            
        fig.update_layout(
            height=500,
            title_text="Review Distribution Analysis",
            template='plotly_white'
        )
        return fig
