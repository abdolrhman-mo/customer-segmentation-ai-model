import streamlit as st
import pandas as pd
from ui.data_manager import load_and_predict_best_model, create_sample_patient

def render_patient_diagnosis_page():
    """Render the Patient Diagnosis page for inputting patient data and getting predictions"""
    st.markdown("## 🔍 Patient Diagnosis")
    
    st.markdown("### 📋 Enter Patient Cell Nucleus Measurements")
    st.markdown("Input the cell nucleus measurements below to get a diagnosis prediction.")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📊 Sample Data"])
    
    with tab1:
        st.markdown("#### Enter measurements manually:")
        
        # Create form for manual input
        with st.form("patient_diagnosis_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Mean Values:**")
                radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.2, step=0.1)
                texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=20.1, step=0.1)
                perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0, step=0.1)
                area_mean = st.number_input("Area Mean", min_value=0.0, max_value=3000.0, value=600.0, step=0.1)
                smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.09, step=0.001)
                compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.08, step=0.001)
                concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.04, step=0.001)
                concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.03, step=0.001)
                symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.18, step=0.001)
                fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.06, step=0.001)
            
            with col2:
                st.markdown("**Standard Error Values:**")
                radius_se = st.number_input("Radius SE", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
                texture_se = st.number_input("Texture SE", min_value=0.0, max_value=10.0, value=1.2, step=0.01)
                perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
                area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
                smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=0.1, value=0.005, step=0.001)
                compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001)
                concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
                concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=0.1, value=0.005, step=0.001)
                symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001)
                fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.1, value=0.003, step=0.001)
            
            # Worst values section
            st.markdown("**Worst Values:**")
            col3, col4 = st.columns(2)
            
            with col3:
                radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=16.0, step=0.1)
                texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=28.0, step=0.1)
                perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=200.0, value=110.0, step=0.1)
                area_worst = st.number_input("Area Worst", min_value=0.0, max_value=3000.0, value=800.0, step=1.0)
                smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=1.0, value=0.13, step=0.001)
            
            with col4:
                compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.18, step=0.001)
                concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=1.0, value=0.10, step=0.001)
                concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.07, step=0.001)
                symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.28, step=0.001)
                fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=1.0, value=0.08, step=0.001)
            
            # Submit button
            submitted = st.form_submit_button("🔍 Get Diagnosis Prediction")
            
            if submitted:
                # Create patient data DataFrame
                patient_data = pd.DataFrame({
                    "radius_mean": [radius_mean],
                    "texture_mean": [texture_mean],
                    "perimeter_mean": [perimeter_mean],
                    "area_mean": [area_mean],
                    "smoothness_mean": [smoothness_mean],
                    "compactness_mean": [compactness_mean],
                    "concavity_mean": [concavity_mean],
                    "concave points_mean": [concave_points_mean],
                    "symmetry_mean": [symmetry_mean],
                    "fractal_dimension_mean": [fractal_dimension_mean],
                    "radius_se": [radius_se],
                    "texture_se": [texture_se],
                    "perimeter_se": [perimeter_se],
                    "area_se": [area_se],
                    "smoothness_se": [smoothness_se],
                    "compactness_se": [compactness_se],
                    "concavity_se": [concavity_se],
                    "concave points_se": [concave_points_se],
                    "symmetry_se": [symmetry_se],
                    "fractal_dimension_se": [fractal_dimension_se],
                    "radius_worst": [radius_worst],
                    "texture_worst": [texture_worst],
                    "perimeter_worst": [perimeter_worst],
                    "area_worst": [area_worst],
                    "smoothness_worst": [smoothness_worst],
                    "compactness_worst": [compactness_worst],
                    "concavity_worst": [concavity_worst],
                    "concave points_worst": [concave_points_worst],
                    "symmetry_worst": [symmetry_worst],
                    "fractal_dimension_worst": [fractal_dimension_worst]
                })
                
                # Get prediction
                display_prediction_results(patient_data)
    
    with tab2:
        st.markdown("#### Use sample data for testing:")
        st.markdown("This uses the sample patient data from your data manager for demonstration.")
        
        if st.button("🧪 Test with Sample Data"):
            sample_patient = create_sample_patient()
            st.success("✅ Sample data loaded successfully!")
            st.dataframe(sample_patient)
            
            # Get prediction
            display_prediction_results(sample_patient)
    
    # Medical disclaimer
    st.markdown("---")
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("""
    **This is a demonstration tool and should NOT be used for actual medical diagnosis.**
    
    - The predictions are based on a machine learning model trained on limited data
    - Always consult with qualified medical professionals for actual diagnosis
    - This tool is for educational and research purposes only
    - The model may have limitations and should not replace medical expertise
    """)

def display_prediction_results(patient_data):
    """Display the prediction results in a nice format"""
    st.markdown("### 🎯 Diagnosis Results")
    
    with st.spinner("🔍 Analyzing patient data..."):
        try:
            # Get prediction using the data manager function
            predictions, probabilities, risk_scores = load_and_predict_best_model(patient_data)
            
            if predictions is not None:
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    # Diagnosis result
                    if predictions[0] == 1:
                        st.error("🚨 **DIAGNOSIS: MALIGNANT**")
                        st.error("High probability of malignant tumor detected")
                    else:
                        st.success("✅ **DIAGNOSIS: BENIGN**")
                        st.success("Low probability of malignant tumor")
                    
                    # Probability
                    prob_percentage = probabilities[0] * 100
                    st.metric("Malignant Probability", f"{prob_percentage:.1f}%")
                
                with col2:
                    # Risk assessment
                    st.info(f"**Risk Level:** {risk_scores[0]}")
                    
                    # Confidence indicator
                    if probabilities[0] >= 0.8:
                        confidence = "Very High"
                        color = "red"
                    elif probabilities[0] >= 0.6:
                        confidence = "High"
                        color = "orange"
                    elif probabilities[0] >= 0.4:
                        confidence = "Medium"
                        color = "yellow"
                    else:
                        confidence = "Low"
                        color = "green"
                    
                    st.metric("Confidence Level", confidence)
                
                # Detailed explanation
                st.markdown("### 📊 Detailed Analysis")
                
                if predictions[0] == 1:
                    st.markdown(f"""
                    **Clinical Interpretation:**
                    - The model predicts a **{prob_percentage:.1f}% probability** of malignancy
                    - This patient falls into the **{risk_scores[0]}** category
                    - **Recommendation:** Further medical evaluation strongly advised
                    - **Next Steps:** Consult with oncologist for comprehensive assessment
                    """)
                else:
                    st.markdown(f"""
                    **Clinical Interpretation:**
                    - The model predicts a **{(1-probabilities[0])*100:.1f}% probability** of benign tumor
                    - This patient falls into the **{risk_scores[0]}** category
                      - **Recommendation:** Regular monitoring may be sufficient
                      - **Next Steps:** Follow standard screening protocols
                    """)
                
                # Export results
                st.markdown("### 💾 Export Results")
                if st.button("📥 Download Results as CSV"):
                    # Create results DataFrame
                    results_df = pd.DataFrame({
                        'Metric': ['Diagnosis', 'Malignant Probability', 'Risk Level', 'Confidence'],
                        'Value': [predictions[0], f"{prob_percentage:.1f}%", risk_scores[0], confidence]
                    })
                    
                    # Convert to CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name="patient_diagnosis_results.csv",
                        mime="text/csv"
                    )
            
            else:
                st.error("❌ Failed to get prediction. Please check if the model is properly loaded.")
                st.info("💡 Make sure you have saved models in the 'models/' folder.")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("💡 This might be due to missing model files or incorrect data format.")
