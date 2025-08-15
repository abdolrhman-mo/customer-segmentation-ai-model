import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Define the models path
models_path = "models"

def load_best_model(patient_data):
    """
    Load the saved best model and make predictions on new patient data
    
    Parameters:
    patient_data: DataFrame with patient features (same columns as training data)
    
    Returns:
    predictions: Diagnosis predictions (1/0)
    probabilities: Diagnosis probabilities
    risk_scores: Risk scores based on threshold
    """
    try:
        # Load the saved components
        best_model = joblib.load(f"{models_path}/best_model.pkl")
        scaler = joblib.load(f"{models_path}/scaler.pkl")
        
        # Load the best threshold
        with open(f"{models_path}/best_threshold.txt", 'r') as f:
            best_threshold = float(f.read().strip())
        
        print(f"✅ Model loaded successfully with threshold: {best_threshold}")
        
        # Preprocess the new data
        patient_scaled = scaler.transform(patient_data)
        
        # Get probabilities
        probabilities = best_model.predict_proba(patient_scaled)[:, 1]
        
        # Apply the optimized threshold
        predictions = [1 if prob >= best_threshold else 0 for prob in probabilities]
        
        # Create risk scores
        risk_scores = []
        for prob in probabilities:
            if prob >= best_threshold:
                if prob >= 0.8:
                    risk_scores.append('Very High Risk (Malignant)')
                elif prob >= 0.6:
                    risk_scores.append('High Risk (Malignant)')
                else:
                    risk_scores.append('Medium Risk (Benign)')
            else:
                if prob <= 0.2:
                    risk_scores.append('Very Low Risk (Benign)')
                else:
                    risk_scores.append('Low Risk (Benign)')
        
        return predictions, probabilities, risk_scores
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def create_sample_patient():
    """
    Create a sample patient DataFrame with all required features
    for testing the prediction function
    """
    sample_patient = pd.DataFrame({
        "radius_mean": [14.2],
        "texture_mean": [20.1],
        "perimeter_mean": [92.0],
        "area_mean": [600.0],
        "smoothness_mean": [0.09],
        "compactness_mean": [0.08],
        "concavity_mean": [0.04],
        "concave points_mean": [0.03],
        "symmetry_mean": [0.18],
        "fractal_dimension_mean": [0.06],
        "radius_se": [0.5],
        "texture_se": [1.2],
        "perimeter_se": [3.0],
        "area_se": [40.0],
        "smoothness_se": [0.005],
        "compactness_se": [0.02],
        "concavity_se": [0.01],
        "concave points_se": [0.005],
        "symmetry_se": [0.02],
        "fractal_dimension_se": [0.003],
        "radius_worst": [16.0],
        "texture_worst": [28.0],
        "perimeter_worst": [110.0],
        "area_worst": [800.0],
        "smoothness_worst": [0.13],
        "compactness_worst": [0.18],
        "concavity_worst": [0.10],
        "concave points_worst": [0.07],
        "symmetry_worst": [0.28],
        "fractal_dimension_worst": [0.08]
    })
    
    return sample_patient

def test_model_loading():
    """
    Test function to verify the model loading and prediction works
    """
    print("\n" + "="*60)
    print("🧪 TESTING MODEL LOADING")
    print("="*60)
    
    # Create sample patient data
    sample_patient = create_sample_patient()
    
    # Make prediction
    predictions, probabilities, risk_scores = load_and_predict_best_model(sample_patient)
    
    if predictions is not None:
        print(f"Sample patient prediction: {predictions[0]}")
        print(f"Diagnosis probability: {probabilities[0]:.3f}")
        print(f"Risk level: {risk_scores[0]}")
    else:
        print("❌ Model loading failed")

# Example usage (uncomment to test)
# if __name__ == "__main__":
#     test_model_loading()


# Example usage (uncomment to test)
# print("\n" + "="*60)
# print("🧪 TESTING MODEL LOADING")
# print("="*60)
# 
# # Create sample patient data (replace with your actual data)
# sample_patient = pd.DataFrame({
#     "radius_mean": [14.2],
#     "texture_mean": [20.1],
#     "perimeter_mean": [92.0],
#     "area_mean": [600.0],
#     "smoothness_mean": [0.09],
#     "compactness_mean": [0.08],
#     "concavity_mean": [0.04],
#     "concave points_mean": [0.03],
#     "symmetry_mean": [0.18],
#     "fractal_dimension_mean": [0.06],
#     "radius_se": [0.5],
#     "texture_se": [1.2],
#     "perimeter_se": [3.0],
#     "area_se": [40.0],
#     "smoothness_se": [0.005],
#     "compactness_se": [0.02],
#     "concavity_se": [0.01],
#     "concave points_se": [0.005],
#     "symmetry_se": [0.02],
#     "fractal_dimension_se": [0.003],
#     "radius_worst": [16.0],
#     "texture_worst": [28.0],
#     "perimeter_worst": [110.0],
#     "area_worst": [800.0],
#     "smoothness_worst": [0.13],
#     "compactness_worst": [0.18],
#     "concavity_worst": [0.10],
#     "concave points_worst": [0.07],
#     "symmetry_worst": [0.28],
#     "fractal_dimension_worst": [0.08]
# })
# 
# # Make prediction
# predictions, probabilities, risk_scores = load_and_predict_best_model(sample_patient)
# 
# if predictions is not None:
#     print(f"Sample patient prediction: {predictions[0]}")
#     print(f"Diagnosis probability: {probabilities[0]:.3f}")
#     print(f"Risk level: {risk_scores[0]}")
