def render_overview():
    st.title("Medicine Effectiveness Analysis")
    
    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model R²", "0.925")
    with col2:
        st.metric("Prediction Accuracy", "92.5%")
    with col3:
        st.metric("MSE", "49.003")

    # Model Performance Comparison
    fig = px.bar(
        x=['RandomForest', 'DecisionTree', 'SVR'],
        y=[0.925, 0.917, 0.854],
        title='Model Performance Comparison'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance
    st.subheader("Feature Importance")
    importance_data = {
        'feature': ['satisfaction_score', 'composition_length', 'side_effects_count'],
        'importance': [0.977890, 0.022069, 0.000041]
    }
    fig = px.bar(importance_data, x='feature', y='importance')
    st.plotly_chart(fig, use_container_width=True)
