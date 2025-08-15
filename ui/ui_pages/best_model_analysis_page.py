import streamlit as st
from ui.data_manager import show_best_model_stats

def render_best_model_analysis_page():
    """Render the Best Model Analysis page using saved performance metrics"""
    st.markdown("## 📊 Best Model Analysis")
    
    # Call the function to display metrics
    metrics = show_best_model_stats()
    
    if metrics is not None:
                
        # Training info
        st.markdown("### 📅 Training Information")
        st.info(f"**Trained on**: {metrics['training_info']['date']}")
        st.info(f"**Dataset Size**: {metrics['training_info']['dataset_size']} samples")
        
        # Key insights
        st.markdown("### 🎯 Key Insights")
        st.success(f"**Maximum Recall Achieved**: {metrics['core_metrics']['recall']:.1%}")
        st.warning(f"**Threshold Trade-off**: Lower threshold ({metrics['best_threshold']:.1f}) = Higher recall")
    
        # Show the best model stats
        st.markdown("### 🏆 Model Performance Metrics")
    
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Algorithm", metrics['algorithm'].upper())
            st.metric("Best Threshold", f"{metrics['best_threshold']:.3f}")
            st.metric("Recall", f"{metrics['core_metrics']['recall']:.4f}")
            st.metric("Precision", f"{metrics['core_metrics']['precision']:.4f}")
        
        with col2:
            st.metric("F1-Score", f"{metrics['core_metrics']['f1_score']:.4f}")
        
    else:
        st.error("❌ No performance metrics found. Please run model training first.")
        st.info("💡 Tip: Use the save_performance_metrics function in your notebook to save model results.")
