"""
Enhanced Healthcare Analytics App
Supports real healthcare datasets and advanced clinical workflows
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import our enhanced data loader and explainability (only these are used)
from real_data_loader import HealthcareDataLoader
from healthcare_explainability import HealthcareExplainabilityEngine

# Set page config
st.set_page_config(
    page_title="Advanced healthcare analytics",
    page_icon="+",
    layout="wide"
)

# Global "Clinical Calm + Human Trust" theme CSS
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont,
                     "Segoe UI", Roboto, Arial, sans-serif;
    }

    h1, h2, h3 {
        font-weight: 600;
        color: #1F4E79;
        letter-spacing: -0.5px;
    }

    h4, h5, h6 {
        color: #2FA4A9;
        font-weight: 500;
    }

    p {
        color: #1C1F23;
        line-height: 1.6;
        font-size: 16px;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #E0E6ED;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }

    /* Enhanced header styling */
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }

    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Card styling for content sections */
    div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Button styling */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1F4E79 0%, #2FA4A9 100%);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.3);
    }

    /* Hide anchor/link icons next to headings */
    [data-testid="stMarkdown"] a[href^="#"] { display: none !important; }
    header a[href^="#"] { display: none !important; }
    .stMarkdown a.headerlink { display: none !important; }
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }
    [data-testid="stMarkdown"] button[aria-label*="link"],
    [data-testid="stMarkdown"] a[aria-label*="link"],
    [data-testid="stMarkdown"] a[aria-label*="Copy"] { display: none !important; }

    /* Info and success boxes styling */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.dataset_type = None
    st.session_state.patient_data = None
if 'explainer_trained' not in st.session_state:
    st.session_state.explainer_trained = False
    st.session_state.explainer_engine = None
    st.session_state.explainer_data = None
if 'mimic_selected_patient' not in st.session_state:
    st.session_state.mimic_selected_patient = None
if 'synthea_selected_patient' not in st.session_state:
    st.session_state.synthea_selected_patient = None
if 'nhanes_selected_participant' not in st.session_state:
    st.session_state.nhanes_selected_participant = None


def _get_clinical_reference_table():
    """Standard clinical reference values for comparison with patient data (used after model is trained)."""
    return pd.DataFrame([
        {"Parameter": "Systolic BP", "Normal range": "90–120 mmHg", "Risk threshold": "≥ 160 mmHg (stage 2)", "Unit": "mmHg"},
        {"Parameter": "Diastolic BP", "Normal range": "60–80 mmHg", "Risk threshold": "≥ 100 mmHg", "Unit": "mmHg"},
        {"Parameter": "Heart rate", "Normal range": "60–100 /min", "Risk threshold": "> 120 or < 50 /min", "Unit": "per min"},
        {"Parameter": "Respiratory rate", "Normal range": "12–20 /min", "Risk threshold": "> 24 or < 10 /min", "Unit": "per min"},
        {"Parameter": "Oxygen saturation (SpO₂)", "Normal range": "≥ 95%", "Risk threshold": "< 92%", "Unit": "%"},
        {"Parameter": "Temperature", "Normal range": "36–37.5 °C", "Risk threshold": "> 38.3 or < 36 °C", "Unit": "°C"},
        {"Parameter": "Glucose (fasting)", "Normal range": "< 100 mg/dL", "Risk threshold": "≥ 126 mg/dL", "Unit": "mg/dL"},
        {"Parameter": "Total cholesterol", "Normal range": "< 200 mg/dL", "Risk threshold": "≥ 240 mg/dL", "Unit": "mg/dL"},
        {"Parameter": "BMI", "Normal range": "18.5–24.9 kg/m²", "Risk threshold": "> 30 or < 18.5 kg/m²", "Unit": "kg/m²"},
    ])


def _normalize_columns(df):
    """Lowercase and normalize column names (spaces -> underscores) for matching."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _detect_custom_format(patient_data):
    """
    Detect if custom upload matches MIMIC, Synthea, or NHANES.
    Returns (dataset_type, data) or (None, None).
    patient_data: dict of filename -> DataFrame
    """
    if not patient_data or not isinstance(patient_data, dict):
        return None, None
    files = list(patient_data.items())
    # Single file: try MIMIC
    if len(files) == 1:
        fname, df = files[0]
        df = _normalize_columns(df)
        if "subject_id" in df.columns and "mortality_30day" in df.columns:
            need_one = ["heart_rate", "systolic_bp", "diastolic_bp", "temperature", "oxygen_saturation", "respiratory_rate", "glucose", "length_of_stay"]
            if any(c in df.columns for c in need_one):
                return "mimic", df
        return None, None
    # Three files: try Synthea or NHANES
    if len(files) != 3:
        return None, None
    dfs = {fname: _normalize_columns(df) for fname, df in files}
    # Synthea: patients (patient_id, birth_date), conditions (patient_id, condition, status), medications (patient_id, medication)
    patients_df = None
    conditions_df = None
    medications_df = None
    for fname, df in dfs.items():
        cols = set(df.columns)
        if "patient_id" in cols and "birth_date" in cols:
            patients_df = df
        elif "patient_id" in cols and "condition" in cols and "status" in cols:
            conditions_df = df
        elif "patient_id" in cols and "medication" in cols:
            medications_df = df
    if patients_df is not None and conditions_df is not None and medications_df is not None:
        # Normalize status to uppercase so ACTIVE is recognized
        conditions_df = conditions_df.copy()
        conditions_df["status"] = conditions_df["status"].astype(str).str.upper().str.strip()
        if conditions_df["status"].eq("ACTIVE").any() or conditions_df.empty:
            return "synthea", {"patients": patients_df, "conditions": conditions_df, "medications": medications_df}
    # NHANES: patients (seqn, age), examinations (seqn, bmi), laboratory (seqn, glucose or cholesterol)
    patients_df = None
    examinations_df = None
    laboratory_df = None
    for fname, df in dfs.items():
        cols = set(df.columns)
        if "seqn" not in cols:
            continue
        if "age" in cols and patients_df is None:
            patients_df = df
        elif "bmi" in cols or "bp_systolic" in cols:
            examinations_df = df
        elif "glucose" in cols or "cholesterol_total" in cols or "hdl_cholesterol" in cols:
            laboratory_df = df
    if patients_df is not None and examinations_df is not None and laboratory_df is not None:
        return "nhanes", {"patients": patients_df, "examinations": examinations_df, "laboratory": laboratory_df}
    return None, None


def main():
    st.title("🏥 Healthcare Analytics Platform")
    st.markdown("""
    *Comprehensive clinical decision support with explainable AI*
    """)
    
    # Dataset Selection
    st.sidebar.header("Dataset Configuration", anchor=False)
    dataset_type = st.sidebar.selectbox(
        "Select Dataset Type",
        ["No dataset type selected", "MIMIC-III Style", "Synthea Healthcare", "NHANES Population", "Upload Custom Data"]
    )

    # Custom upload: show file uploader as soon as "Upload Custom Data" is selected (no button needed)
    if dataset_type == "Upload Custom Data":
        st.sidebar.caption("Upload one or more CSV files below.")
        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV file(s)",
            accept_multiple_files=True,
            type=["csv"],
            key="custom_csv_uploader"
        )
        if uploaded_files:
            st.session_state.patient_data = {}
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    st.session_state.patient_data[file.name] = df
                except Exception as e:
                    st.sidebar.error(f"Could not read {file.name}: {e}")
            if st.session_state.patient_data:
                st.session_state.dataset_type = "custom"
                st.session_state.data_loaded = True
                st.session_state.explainer_trained = False
                st.sidebar.success("Custom data loaded. View dataset analysis below.")
    
    # Auto-load sample data for demonstration
    if not st.session_state.data_loaded and dataset_type != "Upload Custom Data":
        with st.spinner("Loading sample healthcare dataset..."):
            loader = HealthcareDataLoader()
            
            if dataset_type == "MIMIC-III Style":
                data = loader.load_mimic_sample(500)
                st.session_state.patient_data = data
                st.session_state.dataset_type = "mimic"
                st.session_state.explainer_trained = False
                st.session_state.mimic_selected_patient = data['subject_id'].iloc[0]
                st.success("Sample MIMIC-III dataset loaded successfully.")
                
            elif dataset_type == "Synthea Healthcare":
                patients, conditions, medications, procedures = loader.load_synthea_data(300)
                st.session_state.patient_data = {
                    'patients': patients,
                    'conditions': conditions,
                    'medications': medications,
                    'procedures': procedures
                }
                st.session_state.dataset_type = "synthea"
                st.session_state.explainer_trained = False
                st.success("Sample Synthea dataset loaded successfully.")
                
            elif dataset_type == "NHANES Population":
                patients, exam_data, lab_data = loader.load_nhanes_style_data(400)
                st.session_state.patient_data = {
                    'patients': patients,
                    'examinations': exam_data,
                    'laboratory': lab_data
                }
                st.session_state.dataset_type = "nhanes"
                st.session_state.explainer_trained = False
                st.success("Sample NHANES dataset loaded successfully.")
            
            st.session_state.data_loaded = True
    
    # Manual load option
    if st.sidebar.button("Load Dataset"):
        with st.spinner("Loading healthcare dataset..."):
            loader = HealthcareDataLoader()
            
            if dataset_type == "MIMIC-III Style":
                data = loader.load_mimic_sample(1000)
                st.session_state.patient_data = data
                st.session_state.dataset_type = "mimic"
                st.session_state.explainer_trained = False
                st.session_state.mimic_selected_patient = data['subject_id'].iloc[0]
                st.success("MIMIC-III style dataset loaded successfully.")
                
            elif dataset_type == "Synthea Healthcare":
                patients, conditions, medications, procedures = loader.load_synthea_data(500)
                st.session_state.patient_data = {
                    'patients': patients,
                    'conditions': conditions,
                    'medications': medications,
                    'procedures': procedures
                }
                st.session_state.dataset_type = "synthea"
                st.session_state.explainer_trained = False
                st.success("Synthea healthcare dataset loaded successfully.")
                
            elif dataset_type == "NHANES Population":
                patients, exam_data, lab_data = loader.load_nhanes_style_data(800)
                st.session_state.patient_data = {
                    'patients': patients,
                    'examinations': exam_data,
                    'laboratory': lab_data
                }
                st.session_state.dataset_type = "nhanes"
                st.session_state.explainer_trained = False
                st.success("NHANES population health dataset loaded successfully.")
            
            st.session_state.data_loaded = True
    
    # Patient selectors for all datasets (sidebar) — drives analytics, risk factors, and clinical explanations
    if st.session_state.data_loaded and st.session_state.dataset_type == "mimic":
        df = st.session_state.patient_data
        patient_ids = list(sorted(df['subject_id'].unique()))
        try:
            if st.session_state.mimic_selected_patient is None or st.session_state.mimic_selected_patient not in patient_ids:
                st.session_state.mimic_selected_patient = patient_ids[0]
        except (TypeError, ValueError):
            st.session_state.mimic_selected_patient = patient_ids[0]
        st.sidebar.header("Patient", anchor=False)
        idx = next((i for i, p in enumerate(patient_ids) if p == st.session_state.mimic_selected_patient), 0)
        st.session_state.mimic_selected_patient = st.sidebar.selectbox(
            "Select patient",
            options=patient_ids,
            format_func=lambda x: f"Patient {x}",
            index=idx,
            key="mimic_patient_selector"
        )
    
    elif st.session_state.data_loaded and st.session_state.dataset_type == "synthea":
        data = st.session_state.patient_data
        patients = data['patients']
        patient_ids = list(sorted(patients['patient_id'].unique()))
        try:
            if st.session_state.synthea_selected_patient is None or st.session_state.synthea_selected_patient not in patient_ids:
                st.session_state.synthea_selected_patient = patient_ids[0]
        except (TypeError, ValueError):
            st.session_state.synthea_selected_patient = patient_ids[0]
        st.sidebar.header("Patient", anchor=False)
        idx = next((i for i, p in enumerate(patient_ids) if p == st.session_state.synthea_selected_patient), 0)
        st.session_state.synthea_selected_patient = st.sidebar.selectbox(
            "Select patient",
            options=patient_ids,
            format_func=lambda x: f"Patient {x}",
            index=idx,
            key="synthea_patient_selector"
        )
    
    elif st.session_state.data_loaded and st.session_state.dataset_type == "nhanes":
        data = st.session_state.patient_data
        participants = data['patients']
        participant_ids = list(sorted(participants['seqn'].unique()))
        try:
            if st.session_state.nhanes_selected_participant is None or st.session_state.nhanes_selected_participant not in participant_ids:
                st.session_state.nhanes_selected_participant = participant_ids[0]
        except (TypeError, ValueError):
            st.session_state.nhanes_selected_participant = participant_ids[0]
        st.sidebar.header("Participant", anchor=False)
        idx = next((i for i, p in enumerate(participant_ids) if p == st.session_state.nhanes_selected_participant), 0)
        st.session_state.nhanes_selected_participant = st.sidebar.selectbox(
            "Select participant",
            options=participant_ids,
            format_func=lambda x: f"Participant {x}",
            index=idx,
            key="nhanes_participant_selector"
        )
    
    elif st.session_state.data_loaded and st.session_state.dataset_type == "custom":
        data = st.session_state.patient_data
        # Find the main data file with patient identifiers
        main_df = None
        id_col = None
        
        # Look for patient identifier columns
        for fname, df in data.items():
            for col in ["patient_id", "subject_id", "id", "seqn", "record_id"]:
                if col in df.columns:
                    main_df = df
                    id_col = col
                    break
            if main_df is not None:
                break
        
        if main_df is not None and id_col is not None:
            ids = list(sorted(main_df[id_col].dropna().unique()))
            if len(ids) > 1000:  # Limit for performance
                ids = ids[:1000]
            
            try:
                current_id = getattr(st.session_state, f"custom_selected_{id_col}", None)
                if current_id is None or current_id not in ids:
                    current_id = ids[0]
            except (TypeError, ValueError, AttributeError):
                current_id = ids[0]
            
            st.sidebar.header("Patient/Record", anchor=False)
            idx = next((i for i, p in enumerate(ids) if p == current_id), 0)
            selected_id = st.sidebar.selectbox(
                f"Select by {id_col}",
                options=ids,
                format_func=lambda x: f"{id_col.upper()}: {x}",
                index=idx,
                key=f"custom_{id_col}_selector"
            )
            # Store in session state
            setattr(st.session_state, f"custom_selected_{id_col}", selected_id)
    
    # Main content area
    if st.session_state.data_loaded:
        display_dataset_analysis()
        display_risk_prediction()
        display_clinical_explanations()
    else:
        display_welcome_screen()

def display_welcome_screen():
    st.header("Welcome to Healthcare Analytics", anchor=False)
    
    st.markdown("""
    Welcome to our comprehensive healthcare analytics solution that combines 
    **real clinical data analysis** with **explainable AI-powered risk assessment**.
    """)
    
    # Key Features
    st.subheader("Platform Capabilities", anchor=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Multi-Dataset Support**")
        st.write("• MIMIC-III ICU patient data with time-series vital signs")
        st.write("• Synthea synthetic healthcare records")
        st.write("• NHANES population health survey data")
        st.write("• Custom dataset upload capability")
        
        st.markdown("**🩺 Patient-Specific Analysis**")
        st.write("• Individual patient selection across all datasets")
        st.write("• Real-time clinical parameter monitoring")
        st.write("• 7-tier risk classification (0-1 scale)")
        st.write("• Evidence-based clinical recommendations")
        
        st.markdown("**🧠 AI-Powered Insights**")
        st.write("• SHAP explainability for transparent predictions")
        st.write("• Gradient Boosting risk models")
        st.write("• Feature importance visualization")
        st.write("• Clinical decision support")
    
    with col2:
        st.markdown("**📈 Advanced Analytics**")
        st.write("• Interactive data visualizations")
        st.write("• Clinical reference value comparisons")
        st.write("• Risk factor breakdown analysis")
        st.write("• Population health trends")
        
        st.markdown("**🎯 Clinical Applications**")
        st.write("• Risk stratification and assessment")
        st.write("• Treatment recommendation support")
        st.write("• Patient monitoring and follow-up")
        st.write("• Population health management")
        
        st.markdown("**🔒 Professional Standards**")
        st.write("• HIPAA-compliant data handling")
        st.write("• Evidence-based clinical thresholds")
        st.write("• Real-world clinical workflows")
        st.write("• Intuitive medical interface")
    
    # Getting Started
    st.subheader("Quick Start Guide", anchor=False)
    st.info("**1. Select a Dataset Type** from the sidebar to begin your analysis\n"
            "**2. Explore Patient Data** through interactive visualizations\n"
            "**3. Train Risk Models** to unlock AI-powered clinical insights\n"
            "**4. Review Recommendations** for patient-specific care guidance")
    
    st.success("💡 **Tip:** Start with sample data to explore features, then upload your own datasets for custom analysis.")

def display_dataset_analysis():
    st.header("📊 Data Analysis", anchor=False)
    
    if st.session_state.dataset_type == "mimic":
        df = st.session_state.patient_data
        st.subheader("MIMIC-III Overview", anchor=False)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Patients", len(df['subject_id'].unique()))
        col2.metric("Avg Length of Stay", f"{df['length_of_stay'].mean():.1f} days")
        col3.metric("Mortality Rate", f"{df['mortality_30day'].mean()*100:.1f}%")
        col4.metric("Readmission Rate", f"{df['readmission_30day'].mean()*100:.1f}%")
        
        # Admission types
        st.subheader("Patient Admissions", anchor=False)
        admission_counts = df['admission_type'].value_counts()
        fig = px.pie(values=admission_counts.values, names=admission_counts.index,
                    title="Admission Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis (patient-specific — uses patient selected in sidebar)
        st.subheader("Vital Signs Timeline", anchor=False)
        selected_subject_id = st.session_state.mimic_selected_patient
        sample_patient = df[df['subject_id'] == selected_subject_id].sort_values('charttime')
        vital_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 'oxygen_saturation']
        
        if not sample_patient.empty:
            # Create comprehensive and clinically themed time series visualization
            fig = go.Figure()
            
            # Clinically meaningful color mapping for vital signs
            vital_colors = {
                'heart_rate': '#E74C3C',          # Heart rate
                'systolic_bp': '#2980B9',        # Systolic blood pressure
                'diastolic_bp': '#5DADE2',       # Diastolic blood pressure
                'temperature': '#F39C12',        # Temperature
                'oxygen_saturation': '#27AE60',  # Oxygen saturation
                'glucose': '#8E44AD',            # Glucose
            }
            
            # Enhanced styling for better visual appeal
            for col in vital_cols:
                if col in sample_patient.columns:
                    color = vital_colors.get(col, '#2980B9')
                    # Get the data
                    x_data = sample_patient['charttime']
                    y_data = sample_patient[col]
                    
                    # Enhanced trace with smooth lines and better markers
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='lines+markers',
                        name=col.replace('_', ' ').title(),
                        line=dict(
                            width=4,
                            color=color,
                            shape='spline',
                            smoothing=1.0
                        ),
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=
                            '<b>%{text}</b><br>' +
                            'Time: %{x} hours<br>' +
                            'Value: %{y:.1f}<br>' +
                            '<extra></extra>',
                        text=[col.replace('_', ' ').title()] * len(x_data)
                    ))
                    
                    # Add reference zones instead of just lines
                    normal_ranges = {
                        'heart_rate': (60, 100),
                        'systolic_bp': (90, 120),
                        'diastolic_bp': (60, 80),
                        'temperature': (36.0, 37.5),
                        'oxygen_saturation': (95, 100)
                    }
                    
                    if col in normal_ranges:
                        normal_min, normal_max = normal_ranges[col]
                        # Add shaded normal range area
                        fig.add_hrect(
                            y0=normal_min, 
                            y1=normal_max,
                            fillcolor=color,
                            opacity=0.1,
                            layer='below',
                            line_width=0,
                            name=f"{col.replace('_', ' ').title()} normal range"
                        )
                        
                        # Add boundary lines
                        fig.add_hline(
                            y=normal_min, 
                            line_dash="dot", 
                            line_color=color, 
                            opacity=0.7,
                            line_width=2
                        )
                        fig.add_hline(
                            y=normal_max, 
                            line_dash="dot", 
                            line_color=color, 
                            opacity=0.7,
                            line_width=2
                        )
            
            # Professional medical-themed layout (dark theme for continuous monitoring)
            fig.update_layout(
                title={
                    'text': f"Patient {selected_subject_id} — vital signs over time",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': '#ffffff'}
                },
                xaxis_title="Time (hours)",
                yaxis_title="Measurement value",
                height=700,
                showlegend=True,
                hovermode='x unified',
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(family="Arial, sans-serif", size=12, color="#ffffff"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0, 0, 0, 0.8)',
                    font=dict(color='#ffffff')
                )
            )
            
            # No background grid lines
            fig.update_xaxes(
                showgrid=False,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#444444',
                tickfont=dict(color='#ffffff'),
                title_font=dict(color='#ffffff')
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='#444444',
                tickfont=dict(color='#ffffff'),
                title_font=dict(color='#ffffff')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add summary statistics
            st.subheader("Vital Signs Summary", anchor=False)
            summary_data = []
            for col in vital_cols:
                if col in sample_patient.columns:
                    values = sample_patient[col]
                    summary_data.append({
                        'Vital Sign': col.replace('_', ' ').title(),
                        'Min': f"{values.min():.1f}",
                        'Max': f"{values.max():.1f}",
                        'Mean': f"{values.mean():.1f}",
                        'Std Dev': f"{values.std():.1f}"
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
        else:
            st.warning("No patient data available for time series analysis.")
    
    elif st.session_state.dataset_type == "synthea":
        data = st.session_state.patient_data
        st.subheader("Synthea Overview", anchor=False)
            
        # Patient demographics
        patients = data['patients']
        conditions = data['conditions']
        medications = data['medications']
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(patients))
        col2.metric("Conditions", len(conditions['condition'].unique()) if not conditions.empty else 0)
        col3.metric("Medications", len(medications['medication'].unique()) if not medications.empty else 0)
            
        # Age distribution (use 'age' column if present, otherwise compute from birth_date)
        age_data = None
        if 'age' in patients.columns and not patients.empty:
            age_data = patients['age'].dropna()
            age_data = age_data[(age_data >= 0) & (age_data <= 120)]
        elif 'birth_date' in patients.columns and not patients.empty:
            try:
                age_series = (datetime.now() - pd.to_datetime(patients['birth_date'])).dt.days / 365.25
                age_data = age_series.dropna()
                age_data = age_data[(age_data >= 0) & (age_data <= 120)]
            except Exception:
                pass
        
        if age_data is not None and len(age_data) > 0:
            st.subheader("Patient age distribution", anchor=False)
            age_values = np.array(age_data.values if hasattr(age_data, 'values') else list(age_data))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor('#F7F9FB')
            fig.patch.set_facecolor('#F7F9FB')
            n, bins, patches = ax.hist(age_values, bins=30, color='#3498db', edgecolor='#2980b9', linewidth=0.5)
            ax.set_xlabel('Age (years)', color='#1C1F23')
            ax.set_ylabel('Number of patients', color='#1C1F23')
            ax.set_title('Patient age distribution', color='#1C1F23')
            ax.tick_params(colors='#1C1F23')
            ax.spines['bottom'].set_color('#1C1F23')
            ax.spines['left'].set_color('#1C1F23')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        elif not patients.empty:
            st.warning("No valid age data available for distribution plot.")
            
        # Common conditions
        if not conditions.empty:
            st.subheader("Most common conditions", anchor=False)
            condition_counts = conditions['condition'].value_counts().head(10)
            fig = px.bar(x=condition_counts.index, y=condition_counts.values,
                        labels={'x': 'Condition', 'y': 'Count'},
                        title="Top 10 conditions")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        # Medication analysis
        if not medications.empty:
            st.subheader("Most common medications", anchor=False)
            med_counts = medications['medication'].value_counts().head(10)
            fig = px.bar(x=med_counts.index, y=med_counts.values,
                        labels={'x': 'Medication', 'y': 'Count'},
                        title="Top 10 medications")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    elif st.session_state.dataset_type == "nhanes":
        data = st.session_state.patient_data
        st.subheader("NHANES Overview", anchor=False)
            
        patients = data['patients']
        examinations = data['examinations']
        laboratory = data['laboratory']
            
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Participants", len(patients))
        col2.metric("Exam Records", len(examinations))
        col3.metric("Lab Results", len(laboratory))
            
        # Age distribution (matplotlib for reliable rendering)
        if 'age' in patients.columns and not patients.empty:
            st.subheader("Participant age distribution", anchor=False)
            age_values = patients['age'].dropna()
            age_values = age_values[(age_values >= 0) & (age_values <= 120)]
            if len(age_values) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_facecolor('#F7F9FB')
                fig.patch.set_facecolor('#F7F9FB')
                ax.hist(age_values, bins=40, color='#3498db', edgecolor='#2980b9', linewidth=0.5)
                ax.set_xlabel('Age (years)', color='#1C1F23')
                ax.set_ylabel('Number of participants', color='#1C1F23')
                ax.set_title('Participant age distribution', color='#1C1F23')
                ax.tick_params(colors='#1C1F23')
                ax.spines['bottom'].set_color('#1C1F23')
                ax.spines['left'].set_color('#1C1F23')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        # BMI distribution (matplotlib)
        if 'bmi' in examinations.columns and not examinations.empty:
            st.subheader("BMI distribution", anchor=False)
            bmi_values = examinations['bmi'].dropna()
            bmi_values = bmi_values[(bmi_values >= 10) & (bmi_values <= 60)]
            if len(bmi_values) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_facecolor('#F7F9FB')
                fig.patch.set_facecolor('#F7F9FB')
                ax.hist(bmi_values, bins=30, color='#2ecc71', edgecolor='#27ae60', linewidth=0.5)
                ax.set_xlabel('Body mass index', color='#1C1F23')
                ax.set_ylabel('Number of participants', color='#1C1F23')
                ax.set_title('Body mass index distribution', color='#1C1F23')
                ax.tick_params(colors='#1C1F23')
                ax.spines['bottom'].set_color('#1C1F23')
                ax.spines['left'].set_color('#1C1F23')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
        # Laboratory values (matplotlib boxplots)
        if not laboratory.empty:
            st.subheader("Laboratory results", anchor=False)
            lab_cols = ['glucose', 'cholesterol_total', 'hdl_cholesterol']
            available_cols = [col for col in lab_cols if col in laboratory.columns]
            if available_cols:
                lab_data_clean = []
                lab_labels = []
                for col in available_cols:
                    vals = laboratory[col].dropna().values
                    if len(vals) > 0:
                        lab_data_clean.append(vals)
                        lab_labels.append(col.replace('_', ' ').title())
                if lab_data_clean:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.set_facecolor('#F7F9FB')
                    fig.patch.set_facecolor('#F7F9FB')
                    bp = ax.boxplot(lab_data_clean, labels=lab_labels, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('#9b59b6')
                        patch.set_alpha(0.8)
                    for whisker in bp['whiskers']:
                        whisker.set_color('white')
                    for cap in bp['caps']:
                        cap.set_color('white')
                    for median in bp['medians']:
                        median.set_color('white')
                        median.set_linewidth(2)
                    ax.set_ylabel('Value', color='#1C1F23')
                    ax.set_title('Laboratory value distributions', color='#1C1F23')
                    ax.tick_params(colors='#1C1F23')
                    ax.spines['bottom'].set_color('#1C1F23')
                    ax.spines['left'].set_color('#1C1F23')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    plt.xticks(rotation=15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    elif st.session_state.dataset_type == "custom":
        # Custom uploads: patient_data is { filename: DataFrame }
        data = st.session_state.patient_data
        if not data:
            st.warning("No custom data loaded. Use the sidebar to upload CSV file(s) and click **Load Dataset**.")
        else:
            st.subheader("Custom uploaded data", anchor=False)
            for fname, df in data.items():
                st.markdown(f"**{fname}** — {len(df):,} rows × {len(df.columns)} columns")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", len(df))
                col2.metric("Columns", len(df.columns))
                col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                with st.expander(f"Preview and stats: {fname}"):
                    st.dataframe(df.head(100), use_container_width=True)
                    st.caption(f"Columns: {list(df.columns)}")
                # Age distribution if column exists
                if "age" in df.columns and pd.api.types.is_numeric_dtype(df["age"]):
                    age_vals = df["age"].dropna()
                    age_vals = age_vals[(age_vals >= 0) & (age_vals <= 120)]
                    if len(age_vals) > 0:
                        st.subheader(f"Age Distribution - {fname}", anchor=False)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.set_facecolor("#F7F9FB")
                        fig.patch.set_facecolor("#F7F9FB")
                        ax.hist(age_vals, bins=min(30, len(age_vals.unique()) or 10), color="#3498db", edgecolor="#2980b9", linewidth=0.5)
                        ax.set_xlabel("Age (years)", color="#1C1F23")
                        ax.set_ylabel("Count", color="#1C1F23")
                        ax.set_title(f"Age distribution — {fname}", color="#1C1F23")
                        ax.tick_params(colors="#1C1F23")
                        ax.spines["bottom"].set_color("#1C1F23")
                        ax.spines["left"].set_color("#1C1F23")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
            st.success("Use **Clinical Explanations** below to inspect individual rows.")

def display_risk_prediction():
    st.header("🩺 Risk Assessment", anchor=False)
    
    st.markdown("""
    **Train a model with SHAP explainability** to get data-driven clinical recommendations. 
    The model identifies which features (vitals, lab values, conditions) contribute most to risk predictions.
    """)
    
    if st.button("Train Risk Prediction Model with SHAP Explainability"):
        with st.spinner("Training model and building SHAP explainer..."):
            try:
                engine = HealthcareExplainabilityEngine()
                train_type = st.session_state.dataset_type
                train_data = st.session_state.patient_data
                # Custom uploads: detect if files match MIMIC, Synthea, or NHANES format
                if train_type == "custom":
                    detected_type, detected_data = _detect_custom_format(train_data)
                    if detected_type is None:
                        st.error(
                            "**Custom data format not recognized.** To train the model, upload one of:\n\n"
                            "• **MIMIC-style (1 CSV):** columns `subject_id`, `mortality_30day`, and at least one of "
                            "`heart_rate`, `systolic_bp`, `diastolic_bp`, `temperature`, `oxygen_saturation`, `respiratory_rate`, `glucose`, `length_of_stay`.\n\n"
                            "• **Synthea-style (3 CSVs):** (1) **patients:** `patient_id`, `birth_date`; "
                            "(2) **conditions:** `patient_id`, `condition`, `status` (use ACTIVE/RESOLVED); "
                            "(3) **medications:** `patient_id`, `medication`.\n\n"
                            "Column names are case-insensitive. See CHATGPT_PROMPT_CUSTOM_DATA.md for examples."
                        )
                        st.stop()
                    train_type = detected_type
                    train_data = detected_data
                success, result = engine.train_and_explain(train_type, train_data)
                if success:
                    st.session_state.explainer_engine = engine
                    st.session_state.explainer_data = result
                    st.session_state.explainer_trained = True
                    # Use detected format for the rest of the app (analysis + clinical explanations)
                    if st.session_state.dataset_type == "custom" and train_type in ("mimic", "synthea", "nhanes"):
                        st.session_state.dataset_type = train_type
                        st.session_state.patient_data = train_data
                    st.success("Model trained successfully. SHAP explainability is now active.")
                    st.info("Go to clinical explanations below and select a patient to see SHAP-based recommendations.")
                else:
                    st.error(f"Training failed: {result}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def display_clinical_explanations():
    st.header("🩺 Clinical Insights", anchor=False)
    
    st.subheader("Patient Analysis", anchor=False)
    
    # For MIMIC, patient is selected in sidebar; all sections use that selection
    if st.session_state.dataset_type == "mimic":
        df = st.session_state.patient_data
        selected_patient = st.session_state.mimic_selected_patient
        
        # Show detailed patient information
        patient_data = df[df['subject_id'] == selected_patient]
        st.subheader(f"Patient {selected_patient} Analysis", anchor=False)
        
        # Clinical summary
        col1, col2, col3 = st.columns(3)
        col1.metric("Admission Type", patient_data['admission_type'].iloc[0])
        col2.metric("Insurance", patient_data['insurance'].iloc[0])
        col3.metric("Ethnicity", patient_data['ethnicity'].iloc[0])
        
        # Risk factor analysis: patient-specific (this patient only)
        st.subheader("Risk factor analysis", anchor=False)
        risk_factors = {
            'Mortality (30-day)': int(patient_data['mortality_30day'].iloc[0]),
            'Readmission (30-day)': int(patient_data['readmission_30day'].iloc[0]),
            'Systolic BP > 160 mmHg': int(patient_data['systolic_bp'].max() > 160),
            'Systolic BP > 180 mmHg': int(patient_data['systolic_bp'].max() > 180),
            'Heart rate > 120 /min': int(patient_data['heart_rate'].max() > 120) if 'heart_rate' in patient_data.columns else 0,
            'SpO2 < 92%': int(patient_data['oxygen_saturation'].min() < 92) if 'oxygen_saturation' in patient_data.columns else 0
        }
        
        # Green to red color mapping based on risk level
        risk_values = list(risk_factors.values())
        colors = []
        for v in risk_values:
            if v == 0:
                colors.append('#27AE60')  # Green - Low risk
            elif v == 1:
                colors.append('#E74C3C')  # Red - High risk
        
        fig = go.Figure(go.Bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            text=[str(v) for v in risk_factors.values()],
            textposition="outside",
            marker_color=colors
        ))
        fig.update_layout(
            title=f"Patient {selected_patient} — risk factor profile",
            yaxis_title="Risk present (1) / absent (0)",
            xaxis_tickangle=-25,
            height=400,
            yaxis=dict(tickvals=[0, 1], ticktext=['No', 'Yes']),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Standard clinical reference values — shown only after "Train Risk ..." has been clicked
        if st.session_state.explainer_trained:
            st.subheader("Clinical Reference Values", anchor=False)
            st.table(_get_clinical_reference_table())
            st.caption("Use these reference values to compare with the patient's values in the recommendations below.")
            st.markdown("---")
        
        # Clinical recommendations (only shown after training)
        if st.session_state.explainer_trained and st.session_state.explainer_engine and st.session_state.explainer_data:
            st.subheader("Clinical Recommendations", anchor=False)
            expl_df, X, X_scaled, y = st.session_state.explainer_data
            match_idx = np.where(expl_df['subject_id'] == selected_patient)[0]
            if len(match_idx) > 0:
                patient_idx = int(match_idx[0])
                contribs, (risk_inc, risk_dec) = st.session_state.explainer_engine.get_shap_explanations(
                    X_scaled, patient_idx)
                if contribs is not None:
                    expl_text = st.session_state.explainer_engine.generate_explanation_recommendations(
                        patient_idx, expl_df, X_scaled, X, risk_inc, risk_dec, 'mimic')
                    st.markdown("---")
                    st.markdown(expl_text)
                else:
                    for rec in generate_clinical_recommendations(patient_data):
                        st.write(f"• {rec}")
            else:
                for rec in generate_clinical_recommendations(patient_data):
                    st.write(f"• {rec}")
        elif st.session_state.explainer_trained:
            st.subheader("Clinical Recommendations", anchor=False)
            st.info("Patient-specific recommendations will appear here after the model analyzes this patient's data.")
            for rec in generate_clinical_recommendations(patient_data):
                st.write(f"• {rec}")
        else:
            st.info("💡 Click **Train Risk Prediction Model with SHAP Explainability** above to unlock patient-specific clinical recommendations based on real clinical data analysis.")
    
    elif st.session_state.dataset_type == "synthea":
        data = st.session_state.patient_data
        patients = data['patients']
        conditions = data['conditions']
        medications = data['medications']
        
        # Use patient selected in sidebar
        selected_patient = st.session_state.synthea_selected_patient
        
        # Show detailed patient information
        patient_info = patients[patients['patient_id'] == selected_patient]
        st.subheader(f"Patient {selected_patient} History", anchor=False)
        
        # Patient demographics
        if not patient_info.empty:
            birth_date = patient_info['birth_date'].iloc[0]
            age = (datetime.now() - pd.to_datetime(birth_date)).days / 365.25
            col1, col2, col3 = st.columns(3)
            col1.metric("Age", f"{age:.1f} years")
            col2.metric("Gender", "Not specified" if 'gender' not in patient_info.columns else patient_info['gender'].iloc[0])
            col3.metric("Status", "Alive" if patient_info['death_date'].iloc[0] is None else "Deceased")
        
        # Conditions analysis
        patient_conditions = conditions[conditions['patient_id'] == selected_patient]
        if not patient_conditions.empty:
            st.subheader("Active conditions", anchor=False)
            active_conditions = patient_conditions[patient_conditions['status'] == 'ACTIVE']['condition']
            for condition in active_conditions:
                st.warning(condition)
        
        # Medications analysis
        patient_medications = medications[medications['patient_id'] == selected_patient]
        if not patient_medications.empty:
            st.subheader("💊 Current Medications", anchor=False)
            current_meds = patient_medications[patient_medications['end_date'].isna()]
            for _, med in current_meds.iterrows():
                st.info(f"• {med['medication']} ({med['dosage']}) - Started: {med['start_date'].date() if hasattr(med['start_date'], 'date') else med['start_date']}")
        
        # Patient-specific risk assessment based on real clinical parameters
        st.subheader("Risk Assessment", anchor=False)
        
        # Calculate risk based on actual clinical factors
        risk_factors = []
        risk_score = 0.0
        
        # Age-based risk
        if not patient_info.empty:
            age = (datetime.now() - pd.to_datetime(patient_info['birth_date'].iloc[0])).days / 365.25
            if age > 65:
                risk_factors.append(("Advanced age (>65)", 0.3))
                risk_score += 0.3
            elif age > 45:
                risk_factors.append(("Middle age (45-65)", 0.1))
                risk_score += 0.1
        
        # Chronic condition risk
        chronic_conditions = ['Hypertension', 'Diabetes', 'Heart Disease', 'COPD', 'Chronic Kidney Disease']
        active_chronic = patient_conditions[
            (patient_conditions['status'] == 'ACTIVE') & 
            (patient_conditions['condition'].isin(chronic_conditions))
        ]
        
        if len(active_chronic) > 0:
            risk_factors.append((f"{len(active_chronic)} chronic conditions", len(active_chronic) * 0.25))
            risk_score += len(active_chronic) * 0.25
        
        # Polypharmacy risk
        current_meds = patient_medications[patient_medications['end_date'].isna()]
        if len(current_meds) >= 5:
            risk_factors.append((f"Polypharmacy ({len(current_meds)} meds)", 0.2))
            risk_score += 0.2
        elif len(current_meds) >= 3:
            risk_factors.append((f"Multiple medications ({len(current_meds)} meds)", 0.1))
            risk_score += 0.1
        
        # High-risk conditions
        high_risk_conditions = ['Cancer', 'Stroke', 'Myocardial Infarction', 'Congestive Heart Failure']
        active_high_risk = patient_conditions[
            (patient_conditions['status'] == 'ACTIVE') & 
            (patient_conditions['condition'].isin(high_risk_conditions))
        ]
        
        if len(active_high_risk) > 0:
            risk_factors.append((f"High-risk condition: {active_high_risk['condition'].iloc[0]}", 0.4))
            risk_score += 0.4
        
        # Determine granular risk level with green-to-red color scheme (0-1 range)
        if risk_score >= 0.9:
            risk_level = "Fatal"
            risk_color = "#8B0000"  # Dark Red
        elif risk_score >= 0.75:
            risk_level = "Very High"
            risk_color = "#E74C3C"  # Red
        elif risk_score >= 0.55:
            risk_level = "High"
            risk_color = "#E67E22"  # Dark Orange
        elif risk_score >= 0.35:
            risk_level = "Moderate"
            risk_color = "#F39C12"  # Orange
        elif risk_score >= 0.15:
            risk_level = "Low"
            risk_color = "#2ECC71"  # Light Green
        elif risk_score > 0.0:
            risk_level = "Very Low"
            risk_color = "#27AE60"  # Green
        else:
            risk_level = "No Risk"
            risk_color = "#1ABC9C"  # Teal
        
        # Display risk assessment
        col1, col2 = st.columns(2)
        col1.metric("Overall Risk Level", risk_level, f"Score: {risk_score:.2f}")
        
        # Risk factors breakdown
        if risk_factors:
            with col2:
                st.write("**Contributing factors:**")
                for factor, score in risk_factors:
                    st.write(f"- {factor} (+{score:.2f})")
        
        # Visual risk indicator
        st.progress(min(risk_score, 1.0))
        st.caption(f"Risk score: {risk_score:.2f} (0.0 = lowest risk, 1.0+ = highest risk)")
        
        # Health risk assessment explanation — shown only after "Train Risk ..." has been clicked
        if st.session_state.explainer_trained:
            st.subheader("Risk Assessment Guide", anchor=False)
            st.info("**7-Tier Risk Classification (0-1 Scale):**")
            st.write("• **0.0**: No Risk - Healthy patient with no significant risk factors")
            st.write("• **0.0-0.15**: Very Low - Minimal risk factors present")
            st.write("• **0.15-0.35**: Low - Some mild risk factors, generally manageable")
            st.write("• **0.35-0.55**: Moderate - Multiple risk factors requiring attention")
            st.write("• **0.55-0.75**: High - Significant risk factors, intervention recommended")
            st.write("• **0.75-0.9**: Very High - Multiple severe risk factors, urgent care needed")
            st.write("• **0.9-1.0**: Fatal - Critical condition requiring immediate intervention")
            
            st.markdown("---")
        
        # Clinical recommendations (only shown after training)
        if st.session_state.explainer_trained and st.session_state.explainer_engine and st.session_state.explainer_data:
            st.subheader("Clinical Recommendations", anchor=False)
            expl_df, X, X_scaled, y = st.session_state.explainer_data
            match_idx = np.where(expl_df['patient_id'] == selected_patient)[0]
            if len(match_idx) > 0:
                patient_idx = int(match_idx[0])
                contribs, (risk_inc, risk_dec) = st.session_state.explainer_engine.get_shap_explanations(
                    X_scaled, patient_idx)
                if contribs is not None:
                    expl_text = st.session_state.explainer_engine.generate_explanation_recommendations(
                        patient_idx, expl_df, X_scaled, X, risk_inc, risk_dec, 'synthea')
                    st.markdown("---")
                    st.markdown(expl_text)
                else:
                    for rec in generate_synthea_recommendations(patient_conditions, patient_medications):
                        st.write(f"• {rec}")
            else:
                for rec in generate_synthea_recommendations(patient_conditions, patient_medications):
                    st.write(f"• {rec}")
        elif st.session_state.explainer_trained:
            st.subheader("Clinical Recommendations", anchor=False)
            st.info("Patient-specific recommendations will appear here after the model analyzes this patient's data.")
            for rec in generate_synthea_recommendations(patient_conditions, patient_medications):
                st.write(f"• {rec}")
        else:
            st.info("💡 Click **Train Risk Prediction Model with SHAP Explainability** above to unlock patient-specific clinical recommendations based on real clinical data analysis.")
    
    elif st.session_state.dataset_type == "nhanes":
        data = st.session_state.patient_data
        patients = data['patients']
        examinations = data['examinations']
        laboratory = data['laboratory']
        
        # Use participant selected in sidebar
        selected_participant = st.session_state.nhanes_selected_participant
        
        # Show detailed participant information
        participant_info = patients[patients['seqn'] == selected_participant]
        exam_info = examinations[examinations['seqn'] == selected_participant]
        lab_info = laboratory[laboratory['seqn'] == selected_participant]
        
        st.subheader(f"Participant {selected_participant} Analysis", anchor=False)
        
        # Demographics
        if not participant_info.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Age", f"{participant_info['age'].iloc[0]:.1f} years")
            col2.metric("Gender", "Male" if participant_info['gender'].iloc[0] == 1 else "Female")
            col3.metric("BMI", f"{exam_info['bmi'].iloc[0]:.1f}" if not exam_info.empty else "N/A")
        
        # Laboratory results visualization
        if not lab_info.empty:
            st.subheader("Lab Results", anchor=False)
            lab_cols = ['glucose', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']
            available_cols = [col for col in lab_cols if col in lab_info.columns]
            
            if available_cols:
                fig = go.Figure()
                # Define color ranges for different lab values (green to red)
                color_ranges = {
                    'glucose': ['#27AE60', '#F39C12', '#E74C3C'],  # Green -> Orange -> Red
                    'cholesterol_total': ['#27AE60', '#F39C12', '#E74C3C'],
                    'hdl_cholesterol': ['#E74C3C', '#F39C12', '#27AE60'],  # Reverse for HDL (higher is better)
                    'ldl_cholesterol': ['#27AE60', '#F39C12', '#E74C3C']
                }
                
                for col in available_cols:
                    value = lab_info[col].iloc[0]
                    
                    # Determine color based on value and clinical thresholds
                    colors = color_ranges.get(col, ['#27AE60', '#F39C12', '#E74C3C'])
                    if col == 'glucose':
                        # Glucose: normal < 100, prediabetes 100-125, diabetes > 126
                        if value < 100:
                            bar_color = colors[0]  # Green
                        elif value < 126:
                            bar_color = colors[1]  # Orange
                        else:
                            bar_color = colors[2]  # Red
                    elif col == 'hdl_cholesterol':
                        # HDL: low < 40, normal 40-60, high > 60 (higher is better)
                        if value > 60:
                            bar_color = colors[2]  # Green (good)
                        elif value > 40:
                            bar_color = colors[1]  # Orange
                        else:
                            bar_color = colors[0]  # Red (bad)
                    else:
                        # For other cholesterol values: lower is better
                        if col in ['cholesterol_total', 'ldl_cholesterol']:
                            if value < 200:
                                bar_color = colors[0]  # Green
                            elif value < 240:
                                bar_color = colors[1]  # Orange
                            else:
                                bar_color = colors[2]  # Red
                        else:
                            bar_color = colors[1]  # Default orange
                    
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={'text': col.replace('_', ' ').title()},
                        domain={'row': 0, 'column': available_cols.index(col)},
                        gauge={
                            'axis': {'range': [None, max(300, value * 1.2)]},
                            'bar': {'color': bar_color},
                            'steps': [
                                {'range': [0, value * 0.5], 'color': "lightgray"},
                            ],
                        }
                    ))
                fig.update_layout(
                    grid={'rows': 1, 'columns': len(available_cols), 'pattern': "independent"},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Health risk assessment explanation — shown only after "Train Risk ..." has been clicked
        if st.session_state.explainer_trained:
            st.subheader("Risk Assessment Guide", anchor=False)
            st.info("**7-Tier Risk Classification (0-1 Scale):**")
            st.write("• **0.0**: No Risk - Healthy individual with normal laboratory values")
            st.write("• **0.0-0.15**: Very Low - Minor deviations from normal ranges")
            st.write("• **0.15-0.35**: Low - Some borderline values requiring monitoring")
            st.write("• **0.35-0.55**: Moderate - Multiple abnormal values, lifestyle changes needed")
            st.write("• **0.55-0.75**: High - Significant abnormalities, medical intervention required")
            st.write("• **0.75-0.9**: Very High - Severe abnormalities, specialist consultation needed")
            st.write("• **0.9-1.0**: Fatal - Critical values requiring emergency care")
            
            st.markdown("---")
        
        # Patient-specific risk assessment based on real clinical parameters
        st.subheader("Risk Assessment", anchor=False)
        
        # Calculate risk based on actual clinical laboratory values
        risk_factors = []
        risk_score = 0.0
        
        # Age-based risk
        if not participant_info.empty:
            age = participant_info['age'].iloc[0]
            if age > 65:
                risk_factors.append(("Advanced age (>65)", 0.3))
                risk_score += 0.3
            elif age > 45:
                risk_factors.append(("Middle age (45-65)", 0.15))
                risk_score += 0.15
            elif age < 18:
                risk_factors.append(("Pediatric age", 0.1))
                risk_score += 0.1
        
        # BMI-based risk
        if not exam_info.empty and 'bmi' in exam_info.columns:
            bmi = exam_info['bmi'].iloc[0]
            if bmi > 35:
                risk_factors.append((f"Severe obesity (BMI {bmi:.1f})", 0.3))
                risk_score += 0.3
            elif bmi > 30:
                risk_factors.append((f"Obesity (BMI {bmi:.1f})", 0.2))
                risk_score += 0.2
            elif bmi > 25:
                risk_factors.append((f"Overweight (BMI {bmi:.1f})", 0.1))
                risk_score += 0.1
            elif bmi < 18.5:
                risk_factors.append((f"Underweight (BMI {bmi:.1f})", 0.15))
                risk_score += 0.15
        
        # Blood pressure risk
        if not exam_info.empty:
            sys_bp = exam_info['bp_systolic'].iloc[0] if 'bp_systolic' in exam_info.columns else None
            dia_bp = exam_info['bp_diastolic'].iloc[0] if 'bp_diastolic' in exam_info.columns else None
            
            if sys_bp is not None:
                if sys_bp >= 180:
                    risk_factors.append((f"Hypertensive crisis ({sys_bp} mmHg)", 0.4))
                    risk_score += 0.4
                elif sys_bp >= 160:
                    risk_factors.append((f"Stage 2 hypertension ({sys_bp} mmHg)", 0.3))
                    risk_score += 0.3
                elif sys_bp >= 140:
                    risk_factors.append((f"Stage 1 hypertension ({sys_bp} mmHg)", 0.2))
                    risk_score += 0.2
            
            if dia_bp is not None and dia_bp >= 100:
                risk_factors.append((f"Diastolic hypertension ({dia_bp} mmHg)", 0.25))
                risk_score += 0.25
        
        # Laboratory value risks
        if not lab_info.empty:
            # Glucose risk
            if 'glucose' in lab_info.columns:
                glucose = lab_info['glucose'].iloc[0]
                if glucose > 126:
                    risk_factors.append((f"Elevated glucose ({glucose:.0f} mg/dL)", 0.3))
                    risk_score += 0.3
                elif glucose > 100:
                    risk_factors.append((f"Impaired fasting glucose ({glucose:.0f} mg/dL)", 0.15))
                    risk_score += 0.15
            
            # Cholesterol risks
            if 'cholesterol_total' in lab_info.columns:
                total_chol = lab_info['cholesterol_total'].iloc[0]
                if total_chol > 240:
                    risk_factors.append((f"High cholesterol ({total_chol:.0f} mg/dL)", 0.25))
                    risk_score += 0.25
                elif total_chol > 200:
                    risk_factors.append((f"Borderline high cholesterol ({total_chol:.0f} mg/dL)", 0.15))
                    risk_score += 0.15
            
            # LDL cholesterol
            if 'ldl_cholesterol' in lab_info.columns:
                ldl = lab_info['ldl_cholesterol'].iloc[0]
                if ldl > 190:
                    risk_factors.append((f"Very high LDL ({ldl:.0f} mg/dL)", 0.3))
                    risk_score += 0.3
                elif ldl > 160:
                    risk_factors.append((f"High LDL ({ldl:.0f} mg/dL)", 0.2))
                    risk_score += 0.2
                elif ldl > 130:
                    risk_factors.append((f"Borderline high LDL ({ldl:.0f} mg/dL)", 0.1))
                    risk_score += 0.1
            
            # Triglycerides
            if 'triglycerides' in lab_info.columns:
                tg = lab_info['triglycerides'].iloc[0]
                if tg > 500:
                    risk_factors.append((f"Very high triglycerides ({tg:.0f} mg/dL)", 0.25))
                    risk_score += 0.25
                elif tg > 200:
                    risk_factors.append((f"High triglycerides ({tg:.0f} mg/dL)", 0.15))
                    risk_score += 0.15
        
        # Gender-specific considerations
        if not participant_info.empty:
            gender = participant_info['gender'].iloc[0]
            if gender == 2:  # Female
                if age > 50:
                    risk_factors.append(("Postmenopausal female", 0.1))
                    risk_score += 0.1
            else:  # Male
                if age > 45:
                    risk_factors.append(("Middle-aged male", 0.1))
                    risk_score += 0.1
        
        # Determine granular risk level with green-to-red color scheme (0-1 range)
        if risk_score >= 0.9:
            risk_level = "Fatal"
            risk_color = "#8B0000"  # Dark Red
        elif risk_score >= 0.75:
            risk_level = "Very High"
            risk_color = "#E74C3C"  # Red
        elif risk_score >= 0.55:
            risk_level = "High"
            risk_color = "#E67E22"  # Dark Orange
        elif risk_score >= 0.35:
            risk_level = "Moderate"
            risk_color = "#F39C12"  # Orange
        elif risk_score >= 0.15:
            risk_level = "Low"
            risk_color = "#2ECC71"  # Light Green
        elif risk_score > 0.0:
            risk_level = "Very Low"
            risk_color = "#27AE60"  # Green
        else:
            risk_level = "No Risk"
            risk_color = "#1ABC9C"  # Teal
        
        # Display risk assessment
        col1, col2 = st.columns(2)
        col1.metric("Overall Risk Level", risk_level, f"Score: {risk_score:.2f}")
        
        # Risk factors breakdown
        if risk_factors:
            with col2:
                st.write("**Contributing factors:**")
                for factor, score in risk_factors[:5]:  # Show top 5 factors
                    st.write(f"- {factor} (+{score:.2f})")
        
        # Visual risk indicator
        st.progress(min(risk_score, 1.0))
        st.caption(f"Risk score: {risk_score:.2f} (0.0 = no risk, 1.0+ = highest risk)")
        
        # Clinical recommendations (only shown after training)
        if st.session_state.explainer_trained and st.session_state.explainer_engine and st.session_state.explainer_data:
            st.subheader("Clinical Recommendations", anchor=False)
            expl_df, X, X_scaled, y = st.session_state.explainer_data
            match_idx = np.where(expl_df['seqn'] == selected_participant)[0]
            if len(match_idx) > 0:
                patient_idx = int(match_idx[0])
                contribs, (risk_inc, risk_dec) = st.session_state.explainer_engine.get_shap_explanations(
                    X_scaled, patient_idx)
                if contribs is not None:
                    expl_text = st.session_state.explainer_engine.generate_explanation_recommendations(
                        patient_idx, expl_df, X_scaled, X, risk_inc, risk_dec, 'nhanes')
                    st.markdown("---")
                    st.markdown(expl_text)
                else:
                    for rec in generate_nhanes_recommendations(participant_info, exam_info, lab_info):
                        st.write(f"• {rec}")
            else:
                for rec in generate_nhanes_recommendations(participant_info, exam_info, lab_info):
                    st.write(f"• {rec}")
        elif st.session_state.explainer_trained:
            st.subheader("Clinical Recommendations", anchor=False)
            st.info("Patient-specific recommendations will appear here after the model analyzes this participant's data.")
            for rec in generate_nhanes_recommendations(participant_info, exam_info, lab_info):
                st.write(f"• {rec}")
        else:
            st.info("💡 Click **Train Risk Prediction Model with SHAP Explainability** above to unlock patient-specific clinical recommendations based on real clinical data analysis.")

    elif st.session_state.dataset_type == "custom":
        data = st.session_state.patient_data
        if not data:
            st.warning("No custom data loaded. Upload CSV file(s) in the sidebar and click **Load Dataset**.")
        else:
            # Use patient selected in sidebar
            selected_id = None
            id_col = None
            main_df = None
            
            # Find the main data file with patient identifiers
            for fname, df in data.items():
                for col in ["patient_id", "subject_id", "id", "seqn", "record_id"]:
                    if col in df.columns:
                        main_df = df
                        id_col = col
                        selected_id = getattr(st.session_state, f"custom_selected_{id_col}", None)
                        break
                if main_df is not None and selected_id is not None:
                    break
            
            if main_df is not None and id_col is not None and selected_id is not None:
                # Show selected patient data
                row = main_df[main_df[id_col] == selected_id]
                if not row.empty:
                    st.subheader(f"Patient Record: {id_col.upper()} {selected_id}", anchor=False)
                    st.dataframe(row.T if len(row) == 1 else row, use_container_width=True)
                    
                    # Show basic analytics for the selected patient
                    st.subheader("Patient Analytics", anchor=False)
                    numeric_cols = row.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        # Show summary statistics
                        summary_data = []
                        for col in numeric_cols:
                            if not pd.isna(row[col].iloc[0]):
                                summary_data.append({
                                    'Parameter': col.replace('_', ' ').title(),
                                    'Value': f"{row[col].iloc[0]:.2f}" if isinstance(row[col].iloc[0], float) else str(row[col].iloc[0])
                                })
                        if summary_data:
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                    
                    # Show categorical data
                    categorical_cols = row.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        st.subheader("Patient Information", anchor=False)
                        for col in categorical_cols:
                            if not pd.isna(row[col].iloc[0]) and str(row[col].iloc[0]).strip() != "":
                                st.write(f"**{col.replace('_', ' ').title()}:** {row[col].iloc[0]}")
                else:
                    st.warning(f"No data found for {id_col}: {selected_id}")
            else:
                # Fallback: show file selection if no patient ID found
                file_names = list(data.keys())
                chosen_file = st.selectbox("Select uploaded file", file_names, key="custom_file_select")
                df = data[chosen_file]
                if df.empty:
                    st.warning(f"**{chosen_file}** is empty.")
                else:
                    st.subheader(f"File Data: {chosen_file}", anchor=False)
                    st.dataframe(df.head(100), use_container_width=True)

def generate_clinical_recommendations(patient_data):
    """Generate personalized clinical recommendations for MIMIC data"""
    recommendations = []
    
    # Blood pressure management
    if patient_data['systolic_bp'].max() > 180:
        recommendations.append("🚨 **Hypertensive Crisis Alert**: Systolic BP >180 mmHg. Immediate antihypertensive therapy required. Monitor for target organ damage. Consider IV antihypertensives in ICU setting.")
    elif patient_data['systolic_bp'].max() > 160:
        recommendations.append("⚠️ **Stage 2 Hypertension**: Systolic BP >160 mmHg. Optimize antihypertensive regimen. Consider adding ACE inhibitor or ARB. Monitor renal function.")
    elif patient_data['systolic_bp'].max() > 140:
        recommendations.append("⚠️ **Stage 1 Hypertension**: Systolic BP >140 mmHg. Lifestyle modifications plus medication if diabetes or CKD present. Follow JNC 8 guidelines.")
    
    # Heart rate evaluation
    if patient_data['heart_rate'].max() > 120:
        recommendations.append("⚠️ **Tachycardia Alert**: Heart rate >120 bpm. Assess for underlying causes (infection, volume loss, arrhythmia). Consider ECG, electrolytes, and cardiac monitoring.")
    elif patient_data['heart_rate'].min() < 50:
        recommendations.append("⚠️ **Bradycardia Alert**: Heart rate <50 bpm. Evaluate for conduction abnormalities. Consider pacemaker evaluation if symptomatic or unstable.")
    
    # Oxygenation status
    if patient_data['oxygen_saturation'].min() < 90:
        recommendations.append("🚨 **Critical Hypoxemia**: O2 sat <90%. Immediate respiratory assessment. Initiate oxygen therapy. Consider ABG, chest imaging, and pulmonology consultation.")
    elif patient_data['oxygen_saturation'].min() < 92:
        recommendations.append("⚠️ **Hypoxemia**: O2 sat <92%. Evaluate respiratory status. Consider supplemental oxygen. Monitor for respiratory deterioration.")
    
    # Temperature management
    if patient_data['temperature'].max() > 38.3:
        recommendations.append("⚠️ **Fever Present**: Temperature >38.3°C. Investigate infectious source. Consider blood cultures, urinalysis, chest X-ray. Initiate antibiotics if sepsis suspected.")
    elif patient_data['temperature'].min() < 36.0:
        recommendations.append("⚠️ **Hypothermia**: Temperature <36.0°C. Assess for metabolic/endocrine causes. Rule out sepsis. Provide warming measures.")
    
    # Glucose control
    if patient_data['glucose'].max() > 180:
        recommendations.append("⚠️ **Hyperglycemia**: Glucose >180 mg/dL. Initiate insulin protocol if diabetic. Monitor for DKA/HHS. Adjust medications as needed.")
    
    # Length of stay concerns
    if patient_data['length_of_stay'].iloc[0] > 7:
        recommendations.append("⚠️ **Extended Hospitalization**: LOS >7 days. Evaluate for complications, infections, or delayed recovery. Consider early mobility, nutrition, and discharge planning.")
    
    # Readmission risk
    if patient_data['readmission_30day'].iloc[0] == 1:
        recommendations.append("⚠️ **High Readmission Risk**: Previous 30-day readmission. Implement transitional care bundle: medication reconciliation, follow-up scheduling, patient education, home services.")
    
    # Mortality risk
    if patient_data['mortality_30day'].iloc[0] == 1:
        recommendations.append("⚠️ **High Mortality Risk**: Discuss goals of care. Consider palliative care consultation. Optimize comfort measures.")
    
    return recommendations if recommendations else ["✅ **Stable Status**: Continue current care plan with routine monitoring. Patient appears hemodynamically stable with no acute concerns."]

def generate_synthea_recommendations(conditions, medications):
    """Generate recommendations for Synthea data"""
    recommendations = []
    
    # Active conditions analysis
    active_conditions = conditions[conditions['status'] == 'ACTIVE']['condition']
    
    # Specific condition recommendations
    if 'Hypertension' in active_conditions.values:
        recommendations.append("⚠️ **Hypertension Management**: Target BP <130/80 mmHg (ACC/AHA 2017). Monitor annually. Consider ACE-I/ARB, thiazide diuretic, CCB. Assess for target organ damage.")
    
    if 'Diabetes' in active_conditions.values:
        recommendations.append("⚠️ **Diabetes Care**: Target HbA1c <7% (ADA guidelines). Annual ophthalmology, nephrology screening. Foot care education. Consider metformin, insulin, GLP-1 RA, SGLT2i.")
    
    if 'Heart Disease' in active_conditions.values:
        recommendations.append("⚠️ **Cardiovascular Disease**: Aspirin (if no contraindications), statin therapy, ACE-I/ARB, beta-blocker. Cardiology follow-up. Lifestyle modifications.")
    
    if 'COPD' in active_conditions.values:
        recommendations.append("⚠️ **COPD Management**: Bronchodilators (LABA/LAMA), inhaled corticosteroids if indicated. Pulmonary rehab. Annual influenza/pneumococcal vaccines. Smoking cessation.")
    
    if 'Stroke' in active_conditions.values:
        recommendations.append("⚠️ **Post-Stroke Care**: Antiplatelet therapy (aspirin/clopidogrel), statin, BP control <130/80. Neurology follow-up. Physical therapy. Assess swallowing function.")
    
    if 'Cancer' in active_conditions.values:
        recommendations.append("⚠️ **Cancer Care**: Oncology follow-up. Palliative care integration. Nutritional support. Screening for recurrence/metastasis. Vaccinations as appropriate.")
    
    if 'Depression' in active_conditions.values:
        recommendations.append("⚠️ **Mental Health**: PHQ-9 screening. Consider SSRIs/SNRIs. Psychotherapy referral. Monitor for suicide risk. Social support.")
    
    # Medication reconciliation
    if len(medications) >= 5:
        recommendations.append("⚠️ **Polypharmacy Alert**: Review medications for potential interactions, duplications, inappropriate medications per Beers criteria. Consider deprescribing.")
    
    # Medication-specific recommendations
    current_meds = medications[medications['end_date'].isna()]  # Currently prescribed
    if 'Lisinopril' in current_meds['medication'].values:
        recommendations.append("⚠️ **ACE Inhibitor Monitoring**: Check renal function, K+ q3-6 months. Watch for cough. Monitor for angioedema.")
    
    if 'Metformin' in current_meds['medication'].values:
        recommendations.append("⚠️ **Metformin Monitoring**: Check eGFR q6-12 months. Hold if eGFR <30. Screen for B12 deficiency.")
    
    if 'Atorvastatin' in current_meds['medication'].values:
        recommendations.append("⚠️ **Statins Monitoring**: Check lipids q3-6 months. Monitor for myopathy. Consider CoQ10 supplementation if muscle symptoms.")
    
    if 'Warfarin' in current_meds['medication'].values:
        recommendations.append("⚠️ **Anticoagulation**: INR monitoring q4 weeks. Target INR 2-3. Watch for bleeding risk. Consider DOACs if poor INR control.")
    
    # Preventive care reminders
    if 'Diabetes' in active_conditions.values:
        recommendations.append("✅ **Annual Diabetes Screenings**: Ophthalmology, microalbuminuria, HbA1c, lipid panel, foot exam.")
    
    if 'Hypertension' in active_conditions.values:
        recommendations.append("✅ **Regular BP Monitoring**: Check at every visit. Home BP monitoring encouraged.")
    
    if 'Heart Disease' in active_conditions.values:
        recommendations.append("✅ **Cardiovascular Prevention**: Lipid panel, HbA1c, CKD screening. Lifestyle modifications.")
    
    return recommendations if recommendations else ["✅ **Well-Controlled**: No active conditions requiring immediate intervention. Continue routine monitoring and preventive care."]

def generate_nhanes_recommendations(participant_info, exam_info, lab_info):
    """Generate recommendations for NHANES data"""
    recommendations = []
    
    # BMI and weight management
    if not exam_info.empty:
        bmi = exam_info['bmi'].iloc[0] if 'bmi' in exam_info.columns else None
        if bmi is not None:
            if bmi > 40:
                recommendations.append("🚨 **Class III Obesity**: BMI >40. Consider bariatric surgery evaluation. Multidisciplinary weight management. Screen for sleep apnea, diabetes, cardiovascular disease.")
            elif bmi > 35:
                recommendations.append("⚠️ **Class II Obesity**: BMI >35. Comprehensive weight management program. Screen for comorbidities. Consider pharmacotherapy.")
            elif bmi > 30:
                recommendations.append("⚠️ **Class I Obesity**: BMI >30. Lifestyle modification: diet, exercise, behavioral therapy. Screen for diabetes, hypertension, dyslipidemia.")
            elif bmi > 25:
                recommendations.append("⚠️ **Overweight**: BMI >25. Weight management counseling. Emphasize healthy eating and physical activity.")
            elif bmi < 18.5:
                recommendations.append("⚠️ **Underweight**: BMI <18.5. Assess for malnutrition, hyperthyroidism, malignancy, eating disorders. Nutritional support.")
        
        # Blood pressure management
        if 'bp_systolic' in exam_info.columns:
            sys_bp = exam_info['bp_systolic'].iloc[0]
            dia_bp = exam_info['bp_diastolic'].iloc[0] if 'bp_diastolic' in exam_info.columns else None
            
            if sys_bp >= 180 or (dia_bp and dia_bp >= 120):
                recommendations.append("🚨 **Hypertensive Crisis**: BP ≥180/120. Immediate medical evaluation required. Rule out secondary causes. Consider emergency intervention.")
            elif sys_bp >= 160 or (dia_bp and dia_bp >= 100):
                recommendations.append("⚠️ **Stage 2 HTN**: BP ≥160/100. Antihypertensive medication indicated. Target <130/80. Consider ACE-I/ARB, thiazide diuretic.")
            elif sys_bp >= 140 or (dia_bp and dia_bp >= 90):
                recommendations.append("⚠️ **Stage 1 HTN**: BP ≥140/90. Lifestyle modification + medication if ASCVD risk ≥10%. Target <130/80.")
            elif sys_bp >= 130 or (dia_bp and dia_bp >= 80):
                recommendations.append("⚠️ **Elevated BP**: 130-139/80-89. Lifestyle modification: salt reduction, weight loss, exercise. Monitor closely.")
    
    # Laboratory value recommendations
    if not lab_info.empty:
        # Glucose management
        if 'glucose' in lab_info.columns:
            glucose = lab_info['glucose'].iloc[0]
            if glucose > 200:
                recommendations.append("🚨 **Critical Hyperglycemia**: Glucose >200 mg/dL. Diabetes likely. Immediate evaluation. Check HbA1c, urine ketones. Consider insulin therapy.")
            elif glucose > 126:
                recommendations.append("⚠️ **Diabetes Range**: Glucose >126 mg/dL fasting. Confirm with HbA1c or OGTT. Start diabetes management: lifestyle, metformin, monitoring.")
            elif glucose > 100:
                recommendations.append("⚠️ **Prediabetes**: Glucose 100-125 mg/dL fasting. Lifestyle intervention: weight loss 5-10%, exercise 150 min/wk. Monitor annually.")
        
        # Lipid management
        if 'cholesterol_total' in lab_info.columns:
            chol_total = lab_info['cholesterol_total'].iloc[0]
            if chol_total > 240:
                recommendations.append("⚠️ **High Total Cholesterol**: >240 mg/dL. Calculate ASCVD risk. Consider statin therapy. Lifestyle: diet, exercise, weight loss.")
        
        if 'ldl_cholesterol' in lab_info.columns:
            ldl = lab_info['ldl_cholesterol'].iloc[0]
            if ldl > 190:
                recommendations.append("⚠️ **Very High LDL**: >190 mg/dL. High-intensity statin recommended. Rule out familial hypercholesterolemia. Genetic testing may be indicated.")
            elif ldl > 160:
                recommendations.append("⚠️ **High LDL**: >160 mg/dL. Moderate-intensity statin if ASCVD risk ≥7.5%. Lifestyle modification.")
            elif ldl > 130:
                recommendations.append("⚠️ **Borderline LDL**: 130-159 mg/dL. Consider statin if ASCVD risk ≥7.5%. Emphasize lifestyle changes.")
        
        if 'hdl_cholesterol' in lab_info.columns:
            hdl = lab_info['hdl_cholesterol'].iloc[0]
            if hdl < 40:
                recommendations.append("⚠️ **Low HDL**: <40 mg/dL (men), <50 (women). Increased ASCVD risk. Focus on exercise, weight loss, smoking cessation.")
        
        if 'triglycerides' in lab_info.columns:
            trig = lab_info['triglycerides'].iloc[0]
            if trig > 500:
                recommendations.append("🚨 **Severe Hypertriglyceridemia**: >500 mg/dL. Pancreatitis risk. Immediate fibrates/nicotinic acid. Diet: fat restriction <15%, alcohol elimination.")
            elif trig > 200:
                recommendations.append("⚠️ **High Triglycerides**: >200 mg/dL. Lifestyle: diet, exercise, weight loss. Consider fibrate therapy if ASCVD risk high.")
    
    # Age-based recommendations
    if not participant_info.empty:
        age = participant_info['age'].iloc[0]
        gender = participant_info['gender'].iloc[0] if 'gender' in participant_info.columns else None
        
        if age > 65:
            recommendations.append("✅ **Geriatric Care**: Annual flu vaccine, pneumococcal vaccine, bone density scan (women 65+, men 70+), colorectal cancer screening until age 75.")
            recommendations.append("⚠️ **Fall Risk Assessment**: Check vision, gait, balance. Review medications for fall risk. Home safety evaluation.")
        elif age > 50:
            recommendations.append("✅ **Adult Screening**: Colorectal cancer screening (colonoscopy, FIT, etc.). Annual physical exam.")
        
        if age > 40:
            recommendations.append("✅ **ASCVD Risk Assessment**: Calculate 10-year ASCVD risk. Consider statin if risk ≥7.5%. Blood pressure, lipid, diabetes screening.")
        
        if age > 18:
            recommendations.append("✅ **Adult Preventive Care**: Annual wellness visit, blood pressure screening, depression screening, tobacco use assessment.")
        
        # Gender-specific recommendations
        if gender == 2 and age >= 21:  # Female, age 21+
            recommendations.append("✅ **Women's Health**: Pap smear every 3-5 years (age 21-65). Mammography screening starting age 40-50. Bone density screening age 65+.")
        elif gender == 1 and age >= 50:  # Male, age 50+
            recommendations.append("✅ **Men's Health**: Prostate cancer screening discussion (PSA). Colorectal cancer screening. Bone density if risk factors present.")
    
    return recommendations if recommendations else ["✅ **Healthy Individual**: No concerning values identified. Continue routine preventive care and healthy lifestyle habits."]

if __name__ == "__main__":
    main()
