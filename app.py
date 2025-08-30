import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

from src.pipeline.predict_pipeline import PredictPipeline, PredictionHelper
from src.exception import CustomException
from src.logger import logging

# Page configuration
st.set_page_config(
    page_title="Rice Plant Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #2E8B57, #228B22);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #2d5016;
    }
    .disease-info {
        background: #fef9e7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        color: #8b6914;
    }
    .model-info {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load prediction pipeline with caching"""
    try:
        return PredictPipeline()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 Rice Plant Disease Detection System</h1>
        <p>AI-Powered Agricultural Disease Classification using DINOv2</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load pipeline
    pipeline = load_pipeline()
    if pipeline is None:
        st.error("Failed to load the prediction model. Please check the model files.")
        return
    
    # Main layout with columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Upload & Prediction")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a rice plant image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a rice plant for disease detection"
        )
        
        # Sample images option
        st.subheader("Or try sample images:")
        helper = PredictionHelper()
        sample_images = helper.get_sample_images()
        
        if sample_images:
            sample_options = ["Select a sample..."] + [f"{img['label']} - {Path(img['image_path']).name}" for img in sample_images]
            selected_sample = st.selectbox("Choose a sample image:", sample_options)
            
            if selected_sample != "Select a sample...":
                # Find the index in the original sample_images list
                for i, img in enumerate(sample_images):
                    if f"{img['label']} - {Path(img['image_path']).name}" == selected_sample:
                        uploaded_file = img['image_path']
                        break
        
        # Process image if uploaded
        if uploaded_file is not None:
            try:
                # Load and display image
                if isinstance(uploaded_file, str):
                    # Sample image path
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Selected Image", width='stretch')
                    image_input = uploaded_file
                else:
                    # Uploaded file
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", width='stretch')
                    image_input = image
                
                # Make prediction
                if st.button("🔍 Analyze Disease", type="primary", width='stretch'):
                    with st.spinner("Analyzing image..."):
                        result = pipeline.predict_single_image(image_input)
                    
                    # Display results in col2
                    with col2:
                        st.header("🎯 Prediction Results")
                        
                        # Main prediction
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Predicted Disease: {result['predicted_class']}</h3>
                            <p><strong>Confidence:</strong> {result['confidence']*100:.1f}%</p>
                            <p><strong>Model:</strong> {result['model_name']} ({result['model_accuracy']:.1f}% accuracy)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence chart
                        st.subheader("📊 Class Probabilities")
                        prob_df = pd.DataFrame(
                            list(result['all_probabilities'].items()),
                            columns=['Disease', 'Probability']
                        )
                        prob_df['Percentage'] = prob_df['Probability'] * 100
                        
                        fig = px.bar(
                            prob_df, 
                            x='Disease', 
                            y='Percentage',
                            title="Prediction Confidence by Class",
                            color='Percentage',
                            color_continuous_scale='RdYlGn'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, width='stretch')
                        
                        # Disease information
                        disease_info = pipeline.get_disease_info(result['predicted_class'])
                        
                        st.subheader("🔬 Disease Information")
                        st.markdown(f"""
                        <div class="disease-info">
                            <h4>{disease_info['full_name']}</h4>
                            <p><strong>Severity Level:</strong> {disease_info['severity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Symptoms
                        with st.expander("🩺 Symptoms", expanded=True):
                            for symptom in disease_info['symptoms']:
                                st.write(f"• {symptom}")
                        
                        # Treatment
                        with st.expander("💊 Treatment", expanded=True):
                            for treatment in disease_info['treatment']:
                                st.write(f"• {treatment}")
                        
                        # Prevention
                        with st.expander("🛡️ Prevention", expanded=True):
                            for prevention in disease_info['prevention']:
                                st.write(f"• {prevention}")
                        
                        # AI Image Analysis
                        if hasattr(pipeline, 'generate_image_analysis'):
                            st.subheader("🤖 AI Image Analysis")
                            with st.spinner("Generating detailed analysis..."):
                                analysis = pipeline.generate_image_analysis(image_input)
                            
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #17a2b8; color: #31708f;">
                                <h4>Detailed Visual Analysis</h4>
                                <p>{analysis}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("💡 Install transformers to enable AI image analysis: `pip install transformers accelerate`")
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    with col2:
        if uploaded_file is None:

            # Dataset information
            st.subheader("📊 Dataset Information")
            
            dataset_info = {
                'Disease Class': ['Bacterial Blight', 'Brown Spot', 'Leaf Smut'],
                'Severity': ['High', 'Medium', 'Medium'],
                'Common Symptoms': [
                    'Water-soaked lesions, yellow margins',
                    'Brown spots with dark borders', 
                    'Black powdery masses on leaves'
                ]
            }
            
            st.table(pd.DataFrame(dataset_info))

if __name__ == "__main__":
    main()
