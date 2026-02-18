#!/usr/bin/env python3
"""
Earthquake Precursor Detection Dashboard - VPS Version
Optimized for deployment on Virtual Private Server

Author: Earthquake Prediction Research Team
Version: 2.1 (VPS Deployment)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import sys

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DASHBOARD_CONFIG, MODEL_METRICS, VALIDATION_PATHS,
    SCANNER_CONFIG, ASSETS_DIR, MODELS_DIR
)

# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG["title"],
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/earthquake.png", width=80)
    st.title("🌍 EQ Precursor")
    st.markdown(f"**Version**: {DASHBOARD_CONFIG['version']}")
    st.markdown("---")
    
    menu = st.radio(
        "📋 Menu",
        [
            "🏠 Home",
            "📊 Model Performance",
            "✅ Validation Results",
            "🔬 Model Comparison",
            "🔍 Prekursor Scanner",
            "📖 Documentation"
        ]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Magnitude Acc", f"{MODEL_METRICS['magnitude_accuracy']:.2f}%")
    st.metric("LOEO Validation", f"{MODEL_METRICS['loeo_magnitude']:.2f}%")
    st.metric("Model Size", f"{MODEL_METRICS['model_size_mb']} MB")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def load_image(path):
    """Load image with caching"""
    if Path(path).exists():
        return Image.open(path)
    return None

def display_images_grid(image_paths, cols=2, captions=None):
    """Display images in a grid"""
    columns = st.columns(cols)
    for i, path in enumerate(image_paths):
        if Path(path).exists():
            with columns[i % cols]:
                img = Image.open(path)
                caption = captions[i] if captions else Path(path).stem
                st.image(img, caption=caption, use_container_width=True)


# =============================================================================
# HOME PAGE
# =============================================================================
if menu == "🏠 Home":
    st.markdown('<div class="main-header">🌍 Earthquake Precursor Detection Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Dashboard untuk deteksi prekursor gempa bumi menggunakan Deep Learning 
    pada data geomagnetik dari stasiun BMKG Indonesia.
    """)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Magnitude Accuracy",
            f"{MODEL_METRICS['magnitude_accuracy']:.2f}%",
            delta="Production Model"
        )
    
    with col2:
        st.metric(
            "🧭 Azimuth Accuracy", 
            f"{MODEL_METRICS['azimuth_accuracy']:.2f}%",
            delta="8 Directions"
        )
    
    with col3:
        st.metric(
            "📊 LOEO Validation",
            f"{MODEL_METRICS['loeo_magnitude']:.2f}%",
            delta="Temporal Generalization"
        )
    
    with col4:
        st.metric(
            "🗺️ LOSO Validation",
            f"{MODEL_METRICS['loso_magnitude']:.2f}%",
            delta="Spatial Generalization"
        )
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Architecture**: EfficientNet-B0
        - Pre-trained on ImageNet
        - Multi-task learning (Magnitude + Azimuth)
        - Optimized for edge deployment
        """)
    
    with col2:
        st.markdown(f"""
        **Performance**:
        - Model Size: {MODEL_METRICS['model_size_mb']} MB
        - Inference Time: {MODEL_METRICS['inference_time_ms']} ms
        - No data leakage verified
        """)
    
    # Status
    st.markdown("---")
    st.markdown("### 📡 System Status")
    
    model_exists = (MODELS_DIR / "best_final_model.pth").exists()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if model_exists:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Not Found")
    
    with col2:
        st.info("🔄 Dashboard Running")
    
    with col3:
        st.warning("⚠️ SSH: Configure in config.py")


# =============================================================================
# MODEL PERFORMANCE
# =============================================================================
elif menu == "📊 Model Performance":
    st.markdown('<div class="section-header">📊 Model Performance</div>', 
                unsafe_allow_html=True)
    
    # Performance Overview
    st.markdown("### Overall Performance")
    
    metrics_df = pd.DataFrame({
        "Metric": ["Magnitude Accuracy", "Azimuth Accuracy", "LOEO Magnitude", 
                   "LOEO Azimuth", "LOSO Magnitude", "LOSO Azimuth"],
        "Value": [
            MODEL_METRICS['magnitude_accuracy'],
            MODEL_METRICS['azimuth_accuracy'],
            MODEL_METRICS['loeo_magnitude'],
            MODEL_METRICS['loeo_azimuth'],
            MODEL_METRICS['loso_magnitude'],
            MODEL_METRICS['loso_azimuth']
        ]
    })
    
    fig = px.bar(metrics_df, x="Metric", y="Value", 
                 title="Model Performance Metrics",
                 color="Value", color_continuous_scale="Blues")
    fig.update_layout(yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Table
    st.markdown("### Validation Comparison")
    
    comparison_df = pd.DataFrame({
        "Validation Method": ["Random Split", "LOEO (10-Fold)", "LOSO (9-Fold)"],
        "Magnitude Acc (%)": [MODEL_METRICS['magnitude_accuracy'], 
                              MODEL_METRICS['loeo_magnitude'],
                              MODEL_METRICS['loso_magnitude']],
        "Azimuth Acc (%)": [MODEL_METRICS['azimuth_accuracy'],
                           MODEL_METRICS['loeo_azimuth'],
                           MODEL_METRICS['loso_azimuth']]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    st.markdown("""
    <div class="success-box">
    <strong>✅ Key Finding:</strong> LOEO dan LOSO validation menunjukkan model 
    memiliki generalisasi yang baik ke event dan stasiun baru, dengan penurunan 
    performa hanya ~1.4% dari random split.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VALIDATION RESULTS
# =============================================================================
elif menu == "✅ Validation Results":
    st.markdown('<div class="section-header">✅ Validation Results</div>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["LOEO Validation", "LOSO Validation", "Grad-CAM"])
    
    with tab1:
        st.markdown("### LOEO (Leave-One-Event-Out) Validation")
        st.markdown("""
        LOEO validation menguji kemampuan model untuk menggeneralisasi ke 
        event gempa yang belum pernah dilihat selama training.
        """)
        
        loeo_path = VALIDATION_PATHS["loeo_results"]
        if loeo_path.exists():
            images = list(loeo_path.glob("*.png"))
            if images:
                display_images_grid([str(p) for p in images[:4]], cols=2)
            else:
                st.info("No LOEO visualization images found")
        else:
            st.warning(f"LOEO results folder not found: {loeo_path}")
        
        # LOEO Summary
        st.markdown("#### LOEO Summary")
        st.markdown(f"""
        - **Mean Magnitude Accuracy**: {MODEL_METRICS['loeo_magnitude']:.2f}% ± 0.96%
        - **Mean Azimuth Accuracy**: {MODEL_METRICS['loeo_azimuth']:.2f}% ± 5.65%
        - **Coefficient of Variation**: 0.99%
        """)
    
    with tab2:
        st.markdown("### LOSO (Leave-One-Station-Out) Validation")
        st.markdown("""
        LOSO validation menguji kemampuan model untuk menggeneralisasi ke 
        stasiun geomagnetik yang belum pernah dilihat selama training.
        """)
        
        loso_path = VALIDATION_PATHS["loso_results"]
        if loso_path.exists():
            images = list(loso_path.glob("*.png"))
            if images:
                display_images_grid([str(p) for p in images[:4]], cols=2)
            else:
                st.info("No LOSO visualization images found")
        else:
            st.warning(f"LOSO results folder not found: {loso_path}")
        
        # LOSO Summary
        st.markdown("#### LOSO Summary")
        st.markdown(f"""
        - **Weighted Magnitude Accuracy**: {MODEL_METRICS['loso_magnitude']:.2f}%
        - **Weighted Azimuth Accuracy**: {MODEL_METRICS['loso_azimuth']:.2f}%
        - **All stations**: >90% magnitude accuracy
        """)
    
    with tab3:
        st.markdown("### Grad-CAM Visualization")
        st.markdown("""
        Grad-CAM menunjukkan area pada spectrogram yang paling berpengaruh 
        terhadap prediksi model.
        """)
        
        gradcam_path = VALIDATION_PATHS["gradcam_results"]
        if gradcam_path.exists():
            images = list(gradcam_path.glob("*.png"))
            if images:
                display_images_grid([str(p) for p in images[:4]], cols=2)
            else:
                st.info("No Grad-CAM images found")
        else:
            st.warning(f"Grad-CAM folder not found: {gradcam_path}")


# =============================================================================
# MODEL COMPARISON
# =============================================================================
elif menu == "🔬 Model Comparison":
    st.markdown('<div class="section-header">🔬 Model Comparison</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Perbandingan antara arsitektur VGG16 dan EfficientNet-B0 untuk 
    deteksi prekursor gempa bumi.
    """)
    
    # Comparison Table
    comparison_data = {
        "Metric": ["Magnitude Accuracy", "Azimuth Accuracy", "Model Size", 
                   "Inference Time", "Parameters", "Recommendation"],
        "VGG16": ["98.68%", "54.93%", "528 MB", "125 ms", "138M", "Research"],
        "EfficientNet-B0": ["94.37%", "57.39%", "20 MB", "50 ms", "5.3M", "Production"]
    }
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='VGG16',
        x=['Magnitude Acc', 'Azimuth Acc'],
        y=[98.68, 54.93],
        marker_color='#3498db'
    ))
    
    fig.add_trace(go.Bar(
        name='EfficientNet-B0',
        x=['Magnitude Acc', 'Azimuth Acc'],
        y=[94.37, 57.39],
        marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title="Model Accuracy Comparison",
        barmode='group',
        yaxis_range=[0, 100]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Paper Figures
    st.markdown("### Publication Figures")
    
    paper_path = VALIDATION_PATHS["paper_figures"]
    if paper_path.exists():
        images = list(paper_path.glob("*.png"))
        if images:
            selected = st.selectbox("Select Figure", [p.name for p in images])
            if selected:
                st.image(str(paper_path / selected), use_container_width=True)
    else:
        st.info("Paper figures not found")


# =============================================================================
# PREKURSOR SCANNER
# =============================================================================
elif menu == "🔍 Prekursor Scanner":
    st.markdown('<div class="section-header">🔍 Prekursor Scanner</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Scanner untuk mendeteksi prekursor gempa bumi dari data geomagnetik.
    """)
    
    tab1, tab2 = st.tabs(["📁 Upload File", "🌐 SSH Mode"])
    
    with tab1:
        st.markdown("### Upload Spectrogram")
        
        uploaded_file = st.file_uploader(
            "Upload spectrogram image (PNG/JPG)",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Spectrogram", 
                        use_container_width=True)
            
            with col2:
                if st.button("🔍 Analyze", type="primary"):
                    with st.spinner("Analyzing..."):
                        try:
                            from prekursor_scanner import get_scanner
                            
                            scanner = get_scanner()
                            image = Image.open(uploaded_file).convert('RGB')
                            result = scanner.predict(image)
                            
                            st.success("Analysis Complete!")
                            
                            # Display results
                            st.markdown("#### Prediction Results")
                            
                            mag = result['magnitude']
                            azi = result['azimuth']
                            
                            st.metric("Magnitude Class", mag['class'], 
                                     f"{mag['confidence']*100:.1f}% confidence")
                            st.metric("Azimuth Class", azi['class'],
                                     f"{azi['confidence']*100:.1f}% confidence")
                            
                            if result['is_precursor']:
                                st.error("⚠️ PRECURSOR DETECTED!")
                            else:
                                st.success("✅ Normal - No precursor detected")
                                
                        except Exception as e:
                            st.error(f"Error: {e}")
    
    with tab2:
        st.markdown("### SSH Mode (Real-time Data)")
        
        st.warning("""
        ⚠️ SSH mode requires configuration in `app/config.py`.
        Set your BMKG server credentials before using this feature.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            station = st.selectbox("Station", SCANNER_CONFIG["stations"])
        
        with col2:
            scan_date = st.date_input("Date", datetime.now() - timedelta(days=1))
        
        if st.button("🔍 Scan via SSH", type="primary"):
            st.info("SSH scanning not configured. Please update config.py with your credentials.")


# =============================================================================
# DOCUMENTATION
# =============================================================================
elif menu == "📖 Documentation":
    st.markdown('<div class="section-header">📖 Documentation</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### About This Dashboard
    
    Dashboard ini dikembangkan untuk mendeteksi prekursor gempa bumi 
    menggunakan analisis spectrogram data geomagnetik dengan Deep Learning.
    
    ### Model Architecture
    
    - **Base Model**: EfficientNet-B0 (pre-trained ImageNet)
    - **Task**: Multi-task classification (Magnitude + Azimuth)
    - **Input**: 224x224 RGB spectrogram images
    - **Output**: 4 magnitude classes + 9 azimuth classes
    
    ### Validation Methods
    
    1. **LOEO (Leave-One-Event-Out)**: Temporal generalization test
    2. **LOSO (Leave-One-Station-Out)**: Spatial generalization test
    3. **Grad-CAM**: Model interpretability visualization
    
    ### References
    
    - Hayakawa & Molchanov (2002) - Seismo-electromagnetics
    - Tan & Le (2019) - EfficientNet architecture
    - Selvaraju et al. (2017) - Grad-CAM visualization
    
    ### Contact
    
    For questions or issues, please contact the research team.
    """)
    
    # System Info
    st.markdown("---")
    st.markdown("### System Information")
    
    import torch
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        - **Streamlit**: {st.__version__}
        - **PyTorch**: {torch.__version__}
        - **CUDA Available**: {torch.cuda.is_available()}
        """)
    
    with col2:
        st.markdown(f"""
        - **Dashboard Version**: {DASHBOARD_CONFIG['version']}
        - **Model**: EfficientNet-B0
        - **Last Updated**: February 2026
        """)


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    f"<center>Earthquake Precursor Detection Dashboard v{DASHBOARD_CONFIG['version']} | "
    f"© 2026 Research Team</center>",
    unsafe_allow_html=True
)
