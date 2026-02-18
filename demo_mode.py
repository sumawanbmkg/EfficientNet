"""
Demo Mode Helper for Professor Demonstration
Quick access to prepared sample cases

Author: Earthquake Prediction Research Team
Date: February 14, 2026
"""

import json
import streamlit as st
from pathlib import Path

def load_demo_cases():
    """Load demo sample cases from JSON"""
    try:
        with open('demo_sample_cases.json', 'r') as f:
            return json.load(f)
    except:
        return None

def show_demo_selector():
    """Show demo case selector in sidebar"""
    demo_data = load_demo_cases()
    
    if demo_data is None:
        return None
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎓 Demo Mode")
    
    demo_enabled = st.sidebar.checkbox("Enable Demo Mode", value=False,
                                       help="Quick access to prepared sample cases for demonstration")
    
    if demo_enabled:
        cases = demo_data['demo_cases']
        case_names = [f"{c['id']}. {c['name']}" for c in cases]
        
        selected_case_name = st.sidebar.selectbox(
            "Select Demo Case",
            options=case_names,
            help="Pre-configured cases for demonstration"
        )
        
        # Get selected case
        case_id = int(selected_case_name.split('.')[0])
        selected_case = next(c for c in cases if c['id'] == case_id)
        
        # Show case info
        with st.sidebar.expander("📋 Case Details", expanded=True):
            st.write(f"**Description:** {selected_case['description']}")
            st.write(f"**Model:** {selected_case['model']}")
            st.write(f"**Station:** {selected_case['station']}")
            st.write(f"**Date:** {selected_case['date']}")
            st.write(f"**Expected:** {selected_case['expected_label']}")
            st.write(f"**Confidence:** {selected_case['expected_confidence']}")
        
        # Show talking points
        with st.sidebar.expander("💡 Talking Points"):
            for point in selected_case['talking_points']:
                st.write(f"• {point}")
        
        return selected_case
    
    return None

def apply_demo_case(demo_case):
    """Apply demo case to session state"""
    if demo_case:
        st.session_state['demo_station'] = demo_case['station']
        st.session_state['demo_date'] = demo_case['date']
        st.session_state['demo_model'] = demo_case['model']
        return True
    return False

def show_demo_sequence():
    """Show recommended demo sequence"""
    demo_data = load_demo_cases()
    
    if demo_data is None:
        return
    
    st.markdown("### 📋 Recommended Demo Sequence")
    
    for step in demo_data['demo_sequence']:
        case = next(c for c in demo_data['demo_cases'] if c['id'] == step['case_id'])
        
        with st.expander(f"Step {step['step']}: {case['name']} ({step['duration']})"):
            st.write(f"**Focus:** {step['focus']}")
            st.write(f"**Station:** {case['station']}")
            st.write(f"**Date:** {case['date']}")
            st.write(f"**Expected:** {case['expected_label']}")
            
            st.markdown("**Talking Points:**")
            for point in case['talking_points']:
                st.write(f"• {point}")

def show_backup_cases():
    """Show backup cases"""
    demo_data = load_demo_cases()
    
    if demo_data is None:
        return
    
    st.markdown("### 🔄 Backup Cases")
    st.info("Use these if primary cases have issues")
    
    for backup in demo_data['backup_cases']:
        st.write(f"• **{backup['station']}** - {backup['date']} ({backup['label']}) - {backup['note']}")
