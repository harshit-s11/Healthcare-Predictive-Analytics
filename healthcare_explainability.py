"""
Healthcare Explainability Module
Provides SHAP-based explainability for clinical recommendations across dataset types.
Uses GradientBoosting + SHAP TreeExplainer for interpretable, data-driven explanations.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class HealthcareExplainabilityEngine:
    """
    SHAP-based explainability for healthcare risk predictions.
    Trains a tree model and generates recommendations from feature contributions.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.explainer = None
        self.label_encoders = {}
        
    def prepare_mimic_features(self, df):
        """Prepare MIMIC data: aggregate vitals per patient."""
        if 'subject_id' not in df.columns:
            return None, None, None, None
            
        numeric_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature', 
                       'oxygen_saturation', 'respiratory_rate', 'glucose', 'length_of_stay']
        agg_dict = {}
        for c in numeric_cols:
            if c in df.columns:
                agg_dict[c] = ['mean', 'max', 'min']
        agg_dict['mortality_30day'] = 'first'
        agg_dict['readmission_30day'] = 'first'
        
        patient_agg = df.groupby('subject_id').agg(agg_dict)
        new_cols = []
        for c in patient_agg.columns:
            if isinstance(c, tuple):
                new_cols.append(f"{c[0]}_{c[1]}" if c[1] else str(c[0]))
            else:
                new_cols.append(str(c))
        patient_agg.columns = new_cols
        patient_agg = patient_agg.reset_index()
        
        # Target columns get renamed to *_first after agg - find them
        mortality_col = 'mortality_30day_first' if 'mortality_30day_first' in patient_agg.columns else 'mortality_30day'
        feature_cols = [c for c in patient_agg.columns if c not in ['subject_id', 'mortality_30day', 'readmission_30day', 
                       'mortality_30day_first', 'readmission_30day_first']]
        X = patient_agg[feature_cols].fillna(patient_agg[feature_cols].median())
        y = patient_agg[mortality_col].astype(int)
        
        return X, y, patient_agg, feature_cols
    
    def prepare_synthea_features(self, patients, conditions, medications):
        """Prepare Synthea data: merge patients with condition/medication counts and real clinical risk factors."""
        patients = patients.copy()
        # Parse birth_date: support DD-MM-YYYY, YYYY-MM-DD, MM-DD-YYYY
        raw = patients['birth_date'].astype(str).str.strip()
        birth = pd.to_datetime(raw, format='%d-%m-%Y', errors='coerce')
        if birth.isna().all():
            birth = pd.to_datetime(raw, format='%Y-%m-%d', errors='coerce')
        if birth.isna().all():
            birth = pd.to_datetime(raw, dayfirst=True, errors='coerce')
        try:
            if birth.isna().all():
                birth = pd.to_datetime(raw, format='mixed', dayfirst=True, errors='coerce')
        except (TypeError, ValueError):
            pass
        patients['age'] = (pd.Timestamp.now() - birth).dt.days / 365.25
        patients['age'] = patients['age'].clip(0, 120)
        
        # Count active conditions
        cond_counts = conditions[conditions['status'] == 'ACTIVE'].groupby('patient_id')['condition'].agg(['count', list])
        cond_counts.columns = ['n_conditions', 'conditions_list']
        
        # Count current medications (not discontinued)
        current_meds = medications[medications['end_date'].isna()]
        med_counts = current_meds.groupby('patient_id').size().reset_index(name='n_medications')
        
        # Identify high-risk conditions
        high_risk_conditions = ['Hypertension', 'Diabetes', 'Heart Disease', 'COPD', 'Cancer', 
                               'Stroke', 'Myocardial Infarction', 'Congestive Heart Failure', 'Chronic Kidney Disease']
        chronic_conditions = conditions[
            (conditions['status'] == 'ACTIVE') & 
            (conditions['condition'].isin(high_risk_conditions))
        ]
        chronic_counts = chronic_conditions.groupby('patient_id')['condition'].agg(['count', 'first'])
        chronic_counts.columns = ['n_chronic_conditions', 'primary_chronic_condition']
        
        # Merge all features
        df = patients.merge(cond_counts, left_on='patient_id', right_index=True, how='left')
        df = df.merge(med_counts, on='patient_id', how='left')
        df = df.merge(chronic_counts, left_on='patient_id', right_index=True, how='left')
        
        # Fill missing values
        df['n_conditions'] = df['n_conditions'].fillna(0).astype(int)
        df['n_medications'] = df['n_medications'].fillna(0).astype(int)
        df['n_chronic_conditions'] = df['n_chronic_conditions'].fillna(0).astype(int)
        
        # Create condition dummy variables for important conditions
        active_cond = conditions[conditions['status'] == 'ACTIVE']
        if not active_cond.empty:
            condition_dummies = active_cond.groupby('patient_id')['condition'].apply(
                lambda x: pd.get_dummies(x).sum()
            ).unstack(fill_value=0)
            condition_dummies.columns = [f"has_{str(c).lower().replace(' ', '_')}" for c in condition_dummies.columns]
            df = df.merge(condition_dummies, left_on='patient_id', right_index=True, how='left')
            for c in condition_dummies.columns:
                df[c] = df[c].fillna(0).astype(int)
            cond_cols = list(condition_dummies.columns)
        else:
            cond_cols = []
        
        # Define feature columns
        feature_cols = ['age', 'n_conditions', 'n_medications', 'n_chronic_conditions'] + cond_cols
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        # Create patient-specific risk target based on real clinical factors
        X = df[feature_cols].fillna(0)
        
        # More sophisticated risk calculation with granular categories
        risk_score = (
            (df['age'] > 65).astype(int) * 0.3 +
            (df['age'] > 45).astype(int) * 0.15 +
            (df['n_chronic_conditions'] >= 1).astype(int) * 0.3 +
            (df['n_chronic_conditions'] >= 2).astype(int) * 0.2 +
            (df['n_medications'] >= 5).astype(int) * 0.25 +
            (df['n_medications'] >= 3).astype(int) * 0.1
        )
        
        # Create multi-class target for granular risk categories (0-1 range)
        y = pd.cut(risk_score, 
                  bins=[-0.1, 0.0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0], 
                  labels=[0, 1, 2, 3, 4, 5, 6],  # No risk to Fatal
                  include_lowest=True)
        # Handle NaN values by filling with 0 (No Risk) and then convert to int
        y = y.fillna(0).astype(int)
        
        return X, y, df, feature_cols
    
    def prepare_nhanes_features(self, patients, examinations, laboratory):
        """Prepare NHANES data: merge demographics, exams, labs with real clinical risk assessment."""
        df = patients.merge(examinations, on='seqn', how='left')
        df = df.merge(laboratory, on='seqn', how='left')
        
        # Define comprehensive feature set with real clinical parameters
        feature_cols = ['age', 'gender', 'bmi', 'bp_systolic', 'bp_diastolic', 
                       'glucose', 'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
        feature_cols = [c for c in feature_cols if c in df.columns]
        
        # Fill missing values with median
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Create patient-specific risk score based on real clinical thresholds
        risk_score = np.zeros(len(df))
        
        # Age-based risk
        risk_score += (df['age'] > 65).fillna(False).astype(int) * 0.3
        risk_score += (df['age'] > 45).fillna(False).astype(int) * 0.15
        
        # BMI-based risk
        if 'bmi' in df.columns:
            risk_score += (df['bmi'] > 35).fillna(False).astype(int) * 0.3
            risk_score += (df['bmi'] > 30).fillna(False).astype(int) * 0.2
            risk_score += (df['bmi'] > 25).fillna(False).astype(int) * 0.1
            risk_score += (df['bmi'] < 18.5).fillna(False).astype(int) * 0.15
        
        # Blood pressure risk
        if 'bp_systolic' in df.columns:
            risk_score += (df['bp_systolic'] >= 180).fillna(False).astype(int) * 0.4
            risk_score += (df['bp_systolic'] >= 160).fillna(False).astype(int) * 0.3
            risk_score += (df['bp_systolic'] >= 140).fillna(False).astype(int) * 0.2
        
        if 'bp_diastolic' in df.columns:
            risk_score += (df['bp_diastolic'] >= 100).fillna(False).astype(int) * 0.25
        
        # Laboratory value risks
        if 'glucose' in df.columns:
            risk_score += (df['glucose'] > 126).fillna(False).astype(int) * 0.3
            risk_score += (df['glucose'] > 100).fillna(False).astype(int) * 0.15
        
        if 'cholesterol_total' in df.columns:
            risk_score += (df['cholesterol_total'] > 240).fillna(False).astype(int) * 0.25
            risk_score += (df['cholesterol_total'] > 200).fillna(False).astype(int) * 0.15
        
        if 'ldl_cholesterol' in df.columns:
            risk_score += (df['ldl_cholesterol'] > 190).fillna(False).astype(int) * 0.3
            risk_score += (df['ldl_cholesterol'] > 160).fillna(False).astype(int) * 0.2
            risk_score += (df['ldl_cholesterol'] > 130).fillna(False).astype(int) * 0.1
        
        if 'triglycerides' in df.columns:
            risk_score += (df['triglycerides'] > 500).fillna(False).astype(int) * 0.25
            risk_score += (df['triglycerides'] > 200).fillna(False).astype(int) * 0.15
        
        # Create multi-class target for granular risk categories (0-1 range)
        y = pd.cut(risk_score, 
                  bins=[-0.1, 0.0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0], 
                  labels=[0, 1, 2, 3, 4, 5, 6],  # No risk to Fatal
                  include_lowest=True)
        # Handle NaN values by filling with 0 (No Risk) and then convert to int
        y = y.fillna(0).astype(int)
        
        return X, y, df, feature_cols
    
    def train_and_explain(self, dataset_type, data):
        """Train model and create SHAP explainer for the given dataset."""
        X, y, df, feature_cols = None, None, None, None
        
        if dataset_type == 'mimic':
            X, y, df, feature_cols = self.prepare_mimic_features(data)
        elif dataset_type == 'synthea':
            X, y, df, feature_cols = self.prepare_synthea_features(
                data['patients'], data['conditions'], data['medications'])
        elif dataset_type == 'nhanes':
            X, y, df, feature_cols = self.prepare_nhanes_features(
                data['patients'], data['examinations'], data['laboratory'])
        
        if X is None or len(X) < 20 or y.nunique() < 2:
            return False, "Insufficient data for model training"
        
        self.feature_names = feature_cols
        X_scaled = self.scaler.fit_transform(X)
        
        # For multi-class classification, we need to handle SHAP differently
        # Convert to binary classification for SHAP compatibility
        if y.nunique() > 2:
            # Create binary target: 0 for low risk (0-2), 1 for high risk (3-6)
            y_binary = (y >= 3).astype(int)
            self.model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
            self.model.fit(X_scaled, y_binary)
            
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model, X_scaled)
            
            # Store original multi-class labels for reporting
            self.original_y = y
        else:
            # Binary classification - use as-is
            self.model = GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42)
            self.model.fit(X_scaled, y)
            
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model, X_scaled)
            
            self.original_y = y
        
        return True, (df, X, X_scaled, self.original_y if hasattr(self, 'original_y') else y)
    
    def get_shap_explanations(self, X_scaled, patient_idx):
        """Get SHAP values and generate explanation text for a specific patient."""
        if not SHAP_AVAILABLE or self.explainer is None:
            return None, ([], [])

        try:
            shap_values = self.explainer.shap_values(X_scaled)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Positive class (high risk) for binary classification
            patient_shap = shap_values[patient_idx]
            base_value = self.explainer.expected_value
            if isinstance(base_value, np.ndarray):
                base_value = base_value[1]

            contributions = list(zip(self.feature_names, patient_shap))
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            risk_increasing = [(f, v) for f, v in contributions if v > 0.01]
            risk_decreasing = [(f, v) for f, v in contributions if v < -0.01]

            return contributions, (risk_increasing, risk_decreasing)
        except Exception as e:
            return None, ([], [])
    
    def _human_friendly_feature_name(self, feat):
        """Convert technical feature names to human-readable labels."""
        name_map = {
            'heart_rate_mean': 'Average Heart Rate',
            'heart_rate_max': 'Peak Heart Rate',
            'heart_rate_min': 'Lowest Heart Rate',
            'systolic_bp_mean': 'Average Systolic Blood Pressure',
            'systolic_bp_max': 'Peak Systolic Blood Pressure',
            'systolic_bp_min': 'Lowest Systolic Blood Pressure',
            'diastolic_bp_mean': 'Average Diastolic Blood Pressure',
            'diastolic_bp_max': 'Peak Diastolic Blood Pressure',
            'diastolic_bp_min': 'Lowest Diastolic Blood Pressure',
            'temperature_mean': 'Average Temperature',
            'temperature_max': 'Peak Temperature',
            'temperature_min': 'Lowest Temperature',
            'oxygen_saturation_mean': 'Average Oxygen Saturation',
            'oxygen_saturation_max': 'Peak Oxygen Saturation',
            'oxygen_saturation_min': 'Lowest Oxygen Saturation',
            'respiratory_rate_mean': 'Average Respiratory Rate',
            'respiratory_rate_max': 'Peak Respiratory Rate',
            'respiratory_rate_min': 'Lowest Respiratory Rate',
            'glucose_mean': 'Average Blood Glucose',
            'glucose_max': 'Peak Blood Glucose',
            'glucose_min': 'Lowest Blood Glucose',
            'length_of_stay_mean': 'Length of Stay',
            'length_of_stay_max': 'Length of Stay (max)',
            'length_of_stay_min': 'Length of Stay (min)',
            'n_conditions': 'Number of Active Conditions',
            'n_medications': 'Number of Medications',
            'bmi': 'Body Mass Index',
            'age': 'Age',
            'bp_systolic': 'Systolic Blood Pressure',
            'bp_diastolic': 'Diastolic Blood Pressure',
            'cholesterol_total': 'Total Cholesterol',
            'hdl_cholesterol': 'HDL Cholesterol',
            'ldl_cholesterol': 'LDL Cholesterol',
        }
        if feat in name_map:
            return name_map[feat]
        if feat.startswith('has_'):
            return f"Condition: {feat.replace('has_', '').replace('_', ' ').title()}"
        return feat.replace('_', ' ').title()
    
    def _impact_level(self, shap_val):
        """Convert SHAP value to human-readable impact level."""
        abs_val = abs(shap_val)
        if abs_val > 0.15:
            return "strong"
        if abs_val > 0.05:
            return "moderate"
        return "slight"
    
    def _get_patient_raw_value(self, df, idx, feat, X_raw):
        """Get raw numeric value for a feature (for threshold comparison)."""
        try:
            if df is not None and feat in df.columns:
                val = df.iloc[idx][feat]
            elif X_raw is not None and feat in X_raw.columns:
                val = X_raw.iloc[idx][feat]
            else:
                return None
            if val is not None and isinstance(val, (int, float)) and not (isinstance(val, float) and np.isnan(val)):
                return float(val)
        except (IndexError, KeyError, TypeError):
            pass
        return None
    
    def _clinical_threshold(self, feat):
        """Return (risk_above, risk_below, unit) for real clinical risk thresholds. Risk if value > risk_above or < risk_below."""
        feat_lower = feat.lower()
        # Systolic BP: hypertensive crisis >= 180, stage 2 >= 160, stage 1 >= 140
        if 'systolic' in feat_lower and 'diastolic' not in feat_lower:
            return (160, None, 'mmHg')  # risk threshold 160 mmHg
        if 'diastolic' in feat_lower or ('bp_' in feat_lower and 'systolic' not in feat_lower):
            return (100, None, 'mmHg')  # risk threshold 100 mmHg
        if 'heart_rate' in feat_lower:
            return (120, 50, 'per min')  # tachycardia > 120, bradycardia < 50
        if 'respiratory' in feat_lower:
            return (24, 10, 'per min')  # tachypnea > 24, bradypnea < 10
        if 'oxygen' in feat_lower:
            return (None, 92, '%')  # hypoxemia < 92%
        if 'temperature' in feat_lower:
            return (38.3, 36.0, '°C')  # fever > 38.3, hypothermia < 36 (assuming °C; MIMIC often in °C)
        if 'glucose' in feat_lower:
            return (126, None, 'mg/dL')  # diabetes threshold fasting
        if 'cholesterol_total' in feat_lower or 'ldl' in feat_lower:
            return (240, None, 'mg/dL')  # high cholesterol
        if 'bmi' in feat_lower:
            return (30, 18.5, 'kg/m²')  # obese > 30, underweight < 18.5
        return (None, None, None)
    
    def generate_explanation_recommendations(self, patient_idx, df, X_scaled, X_raw, 
                                             risk_increasing, risk_decreasing, dataset_type):
        """Generate human-friendly clinical recommendations from SHAP explanations."""
        pred_proba = self.model.predict_proba(X_scaled[patient_idx:patient_idx+1])[0][1]
        pred_class = self.model.predict(X_scaled[patient_idx:patient_idx+1])[0]
        
        sections = []
        
        # --- Executive Summary ---
        risk_label = "**High Risk**" if pred_class == 1 else "**Low Risk**"
        risk_pct = f"{pred_proba:.0%}" if pred_proba >= 0.1 else "under 10%"
        confidence = "high confidence" if pred_proba > 0.7 or pred_proba < 0.3 else "moderate confidence"
        
        sections.append("## Summary")
        sections.append("")
        sections.append(f"The AI model predicts this patient has **{risk_label}** "
                       f"(estimated probability: {risk_pct}) with {confidence}.")
        sections.append("")
        
        # --- Plain English Explanation ---
        sections.append("## Why This Prediction?")
        sections.append("")
        sections.append("The model analyzed this patient's data and identified the following factors that influenced the risk assessment:")
        sections.append("")
        
        if risk_increasing:
            sections.append("### Factors That Increase Risk")
            sections.append("")
            for i, (feat, val) in enumerate(risk_increasing[:5], 1):
                feat_display = self._human_friendly_feature_name(feat)
                patient_val = self._get_patient_value(df, patient_idx, feat, X_raw)
                raw_val = self._get_patient_raw_value(df, patient_idx, feat, X_raw)
                impact = self._impact_level(val)
                risk_above, risk_below, unit = self._clinical_threshold(feat)
                threshold_note = ""
                if raw_val is not None and (risk_above is not None or risk_below is not None):
                    if risk_above is not None and raw_val > risk_above:
                        threshold_note = f" (above risk threshold {risk_above} {unit})"
                    elif risk_below is not None and raw_val < risk_below:
                        threshold_note = f" (below risk threshold {risk_below} {unit})"
                
                line = f"**{i}. {feat_display}**"
                if patient_val is not None:
                    line += f" — Patient's value: **{patient_val}**{threshold_note}"
                line += f" *(has {impact} impact on raising risk)*"
                sections.append(line)
                sections.append("")
                
                rec = self._feature_to_recommendation(feat, patient_val, raw_val, increasing=True)
                if rec:
                    sections.append(f"   > **Recommendation:** {rec}")
                    sections.append("")
            sections.append("")
        
        if risk_decreasing:
            sections.append("### Factors That Decrease Risk")
            sections.append("")
            sections.append("These factors work in the patient's favor:")
            sections.append("")
            for i, (feat, val) in enumerate(risk_decreasing[:3], 1):
                feat_display = self._human_friendly_feature_name(feat)
                patient_val = self._get_patient_value(df, patient_idx, feat, X_raw)
                
                line = f"**{i}. {feat_display}**"
                if patient_val is not None:
                    line += f" — Patient's value: {patient_val}"
                sections.append(f"- {line}")
            sections.append("")
        
        # --- Clinical Action Items ---
        sections.append("## Suggested Next Steps")
        sections.append("")
        if risk_increasing:
            top_factor = self._human_friendly_feature_name(risk_increasing[0][0])
            sections.append(f"1. **Focus on {top_factor}** — Address this factor first as it has the largest impact on risk.")
            sections.append("")
            sections.append("2. **Review** the specific recommendations listed above for each concerning factor.")
            sections.append("")
            sections.append("3. **Monitor** — Consider more frequent follow-up given the identified risk factors.")
        else:
            sections.append("1. **Continue routine care** — No major risk-increasing factors were identified.")
            sections.append("")
            sections.append("2. **Maintain** current monitoring and preventive measures.")
        
        sections.append("")
        sections.append("---")
        sections.append("*This explanation is generated using SHAP (SHapley Additive exPlanations), a method that quantifies each factor's contribution to the prediction.*")
        
        return "\n".join(sections)
    
    def _get_patient_value(self, df, idx, feat, X_raw):
        """Get patient's value for a feature for display with units where appropriate."""
        try:
            val = None
            if df is not None and feat in df.columns:
                val = df.iloc[idx][feat]
            elif X_raw is not None and feat in X_raw.columns:
                val = X_raw.iloc[idx][feat]
            if val is not None and isinstance(val, (int, float)):
                feat_lower = feat.lower()
                if 'heart_rate' in feat_lower or 'respiratory' in feat_lower:
                    return f"{val:.0f} per min"
                if 'systolic' in feat_lower or 'diastolic' in feat_lower or 'bp_' in feat_lower:
                    return f"{val:.0f} mmHg"
                if 'temperature' in feat_lower:
                    return f"{val:.1f}°F"
                if 'oxygen' in feat_lower:
                    return f"{val:.1f}%"
                if 'glucose' in feat_lower:
                    return f"{val:.0f} mg/dL"
                if 'cholesterol' in feat_lower:
                    return f"{val:.0f} mg/dL"
                if 'bmi' in feat_lower:
                    return f"{val:.1f} kg/m²"
                if 'age' in feat_lower:
                    return f"{val:.1f} years"
                return f"{val:.2f}"
            elif val is not None:
                return str(val)
        except (IndexError, KeyError, TypeError):
            pass
        return None
    
    def _feature_to_recommendation(self, feat, value_display, raw_value, increasing=True):
        """Map SHAP-identified feature to actionable clinical recommendation with real thresholds."""
        feat_lower = feat.lower()
        risk_above, risk_below, unit = self._clinical_threshold(feat)
        prefix = ""
        if raw_value is not None and unit and increasing:
            raw_str = f"{raw_value:.0f}" if raw_value == int(raw_value) else f"{raw_value:.1f}"
            if risk_above is not None and raw_value > risk_above:
                prefix = f"Value {raw_str} {unit} exceeds risk threshold ({risk_above} {unit}). "
            elif risk_below is not None and raw_value < risk_below:
                prefix = f"Value {raw_str} {unit} is below safe threshold ({risk_below} {unit}). "
        if increasing:
            if 'heart_rate' in feat_lower or 'hr_' in feat_lower:
                return prefix + "Monitor heart rate; consider cardiac evaluation if persistently elevated (risk if > 120 or < 50 per min)."
            if 'systolic' in feat_lower and 'diastolic' not in feat_lower:
                return prefix + "Blood pressure management recommended; hypertensive crisis threshold 180 mmHg, stage 2 threshold 160 mmHg."
            if 'diastolic' in feat_lower or ('bp_' in feat_lower and 'systolic' not in feat_lower):
                return prefix + "Diastolic BP management; risk threshold 100 mmHg."
            if 'oxygen' in feat_lower or 'o2' in feat_lower:
                return prefix + "Assess respiratory status; hypoxemia risk below 92% SpO2. Evaluate need for supplemental oxygen."
            if 'glucose' in feat_lower:
                return prefix + "Blood glucose monitoring; diabetes threshold 126 mg/dL fasting. Consider screening or medication review."
            if 'cholesterol' in feat_lower:
                return prefix + "Lipid panel review; high cholesterol threshold 240 mg/dL. Consider statin therapy per guidelines."
            if 'bmi' in feat_lower:
                return prefix + "Weight management; obesity threshold 30 kg/m². Dietary and exercise counseling."
            if 'temperature' in feat_lower:
                return prefix + "Fever > 38.3°C or hypothermia < 36°C indicates risk. Monitor and treat accordingly."
            if 'respiratory' in feat_lower:
                return prefix + "Respiratory rate risk if > 24 or < 10 per min. Assess respiratory status."
            if 'n_conditions' in feat_lower:
                return "Multiple conditions present—ensure care coordination and medication reconciliation."
            if 'n_medications' in feat_lower:
                return "Polypharmacy review; consider deprescribing where appropriate."
            if 'has_diabetes' in feat_lower:
                return "Ensure annual eye exam, foot exam, and A1C monitoring."
            if 'has_hypertension' in feat_lower:
                return "Monitor blood pressure at every visit; optimize antihypertensive therapy."
            if 'has_asthma' in feat_lower or 'has_copd' in feat_lower:
                return "Ensure inhaler technique review; assess control and consider step-up if needed."
            if 'has_heart' in feat_lower:
                return "Cardiac follow-up; ensure appropriate antiplatelet/statin therapy."
        return ""
