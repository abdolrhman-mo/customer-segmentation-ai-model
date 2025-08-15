import streamlit as st

# Import our modular components
from ui.ui_pages.best_model_analysis_page import render_best_model_analysis_page
# from ui.ui_pages.patient_diagnosis_page import render_patient_diagnosis_page
from ui.ui_pages.technical_implementation_page import render_technical_implementation_page

# Page config
st.set_page_config(
    page_title="Breast Cancer Diagnosis",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .recall-highlight {
        background: linear-gradient(90deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 Breast Cancer Diagnosis App</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio("Choose Section:", [
    "📊 Best Model Analysis",
    # "🔍 Patient Diagnosis",
    "⚙️ Technical Implementation"
])

# Page routing
if page == "📊 Best Model Analysis":
    render_best_model_analysis_page()
elif page == "🔍 Patient Diagnosis":
    render_patient_diagnosis_page()
elif page == "⚙️ Technical Implementation":
    render_technical_implementation_page()
