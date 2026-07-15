"""
Real Healthcare Data Loader
Handles loading from various legitimate healthcare data sources
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HealthcareDataLoader:
    """
    Load and process real healthcare datasets for predictive analytics
    """
    
    def __init__(self):
        self.datasets = {
            'mimic': 'MIMIC-III Critical Care Database',
            'synthea': 'Synthetic Health Data',
            'nhanes': 'NHANES Survey Data',
            'custom': 'Custom Healthcare Dataset'
        }
    
    def load_mimic_sample(self, sample_size=1000):
        """
        Load sample MIMIC-III style data (simulated structure)
        In practice, you'd connect to actual MIMIC database
        """
        # Note: Avoid emojis here to prevent UnicodeEncodeError on some Windows consoles
        print("Loading MIMIC-III style MIMIC dataset...")
        
        # Simulate realistic MIMIC structure
        np.random.seed(42)
        n_patients = sample_size
        
        data = {
            'subject_id': range(1, n_patients + 1),
            'hadm_id': range(1001, 1001 + n_patients),
            'admittime': [datetime(2023, 1, 1) + timedelta(hours=np.random.randint(0, 8760)) 
                         for _ in range(n_patients)],
            'dischtime': [datetime(2023, 1, 5) + timedelta(hours=np.random.randint(0, 168)) 
                         for _ in range(n_patients)],
            'admission_type': np.random.choice(['EMERGENCY', 'ELECTIVE', 'URGENT'], n_patients),
            'insurance': np.random.choice(['Medicare', 'Private', 'Medicaid'], n_patients),
            'language': np.random.choice(['ENGLISH', 'SPANISH', 'OTHER'], n_patients),
            'marital_status': np.random.choice(['MARRIED', 'SINGLE', 'WIDOWED', 'DIVORCED'], n_patients),
            'ethnicity': np.random.choice([
                'WHITE', 'BLACK/AFRICAN AMERICAN', 'HISPANIC/LATINO', 
                'ASIAN', 'OTHER'
            ], n_patients)
        }
        
        # Generate time-series vital signs for each patient
        vital_signs = self._generate_vital_signs_timeseries(n_patients, 24)  # 24 hours of data
        
        # Combine patient info with vital signs
        df = pd.DataFrame(data)
        df = df.merge(vital_signs, on='subject_id', how='left')
        
        # Add clinical outcomes
        df['mortality_30day'] = self._generate_mortality_risk(df)
        df['readmission_30day'] = self._generate_readmission_risk(df)
        df['length_of_stay'] = self._generate_los(df)
        
        print(f"Loaded {len(df)} patient records with {len(df.columns)} features")
        return df
    
    def load_synthea_data(self, sample_size=1000):
        """
        Load Synthea synthetic healthcare data structure
        """
        print("Loading Synthea-style synthetic data...")
        
        np.random.seed(123)
        n_patients = sample_size
        
        # Patient demographics
        data = {
            'patient_id': range(1, n_patients + 1),
            'birth_date': [datetime(1950, 1, 1) + timedelta(days=np.random.randint(0, 25000)) 
                          for _ in range(n_patients)],
            'death_date': [None if np.random.random() > 0.15 else 
                          datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1825))
                          for _ in range(n_patients)],
            'ssn': [f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}" 
                   for _ in range(n_patients)],
            'drivers': [f"D{np.random.randint(1000000, 9999999)}" if np.random.random() > 0.2 else None 
                       for _ in range(n_patients)],
            'passport': [f"P{np.random.randint(10000000, 99999999)}" if np.random.random() > 0.7 else None 
                        for _ in range(n_patients)]
        }
        
        df = pd.DataFrame(data)
        
        # Add age column for visualization (computed from birth_date)
        df['age'] = (datetime.now() - pd.to_datetime(df['birth_date'])).dt.days / 365.25
        df['age'] = df['age'].clip(0, 120)  # Keep ages in valid range
        
        # Add conditions, medications, procedures (simplified)
        conditions = self._generate_conditions(df)
        medications = self._generate_medications(df)
        procedures = self._generate_procedures(df)
        
        print(f"Generated {len(df)} synthetic patient records")
        print(f"Conditions: {conditions['condition'].nunique()} unique conditions")
        print(f"Medications: {medications['medication'].nunique()} unique medications")
        print(f"Procedures: {procedures['procedure'].nunique()} unique procedures")
        
        return df, conditions, medications, procedures
    
    def load_nhanes_style_data(self, sample_size=1000):
        """
        Load NHANES-style survey data
        """
        print("Loading NHANES-style population health data...")
        
        np.random.seed(456)
        n_patients = sample_size
        
        data = {
            'seqn': range(10001, 10001 + n_patients),  # NHANES sequence numbers
            'age': np.random.normal(45, 18, n_patients),
            'gender': np.random.choice([1, 2], n_patients),  # 1=Male, 2=Female
            'race': np.random.choice([1, 2, 3, 4, 5], n_patients),  # NHANES race codes
            'education': np.random.choice([1, 2, 3, 4, 5, 7, 9], n_patients),  # Education levels
            'marital_status': np.random.choice([1, 2, 3, 4, 5, 6, 77, 99], n_patients),
            'annual_income': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 77, 99], n_patients)
        }
        
        df = pd.DataFrame(data)
        
        # Add health examination data
        exam_data = self._generate_examination_data(df)
        lab_data = self._generate_laboratory_data(df)
        
        print(f"✅ Loaded {len(df)} NHANES-style records")
        return df, exam_data, lab_data
    
    def _generate_vital_signs_timeseries(self, n_patients, hours):
        """Generate realistic time-series vital signs"""
        records = []
        
        for patient_id in range(1, n_patients + 1):
            # Patient baseline characteristics
            age = np.random.normal(65, 15)
            is_critical = np.random.random() < 0.15
            
            for hour in range(hours):
                # Simulate vital signs with realistic variation
                hr = np.random.normal(85 if not is_critical else 110, 15)
                sys_bp = np.random.normal(125 if not is_critical else 150, 20)
                dia_bp = sys_bp * np.random.normal(0.65, 0.05)
                temp = np.random.normal(98.6, 0.8)
                o2_sat = np.random.normal(97 if not is_critical else 92, 3)
                resp_rate = np.random.normal(16 if not is_critical else 22, 4)
                
                records.append({
                    'subject_id': patient_id,
                    'charttime': hour,
                    'heart_rate': max(40, min(180, hr)),
                    'systolic_bp': max(80, min(220, sys_bp)),
                    'diastolic_bp': max(50, min(140, dia_bp)),
                    'temperature': max(95, min(104, temp)),
                    'oxygen_saturation': max(85, min(100, o2_sat)),
                    'respiratory_rate': max(8, min(40, resp_rate)),
                    'glucose': np.random.normal(100, 25),
                    'sodium': np.random.normal(140, 4),
                    'potassium': np.random.normal(4.2, 0.5)
                })
        
        return pd.DataFrame(records)
    
    def _generate_mortality_risk(self, df):
        """Generate realistic mortality risk with varied profiles"""
        n_patients = len(df)
        risk_scores = np.zeros(n_patients)
        
        # Create distinct risk groups
        # Low risk group (40%): 0-0.3 score
        low_risk_count = int(n_patients * 0.4)
        # Moderate risk group (35%): 0.3-0.6 score
        moderate_risk_count = int(n_patients * 0.35)
        # High risk group (25%): 0.6-1.0 score
        high_risk_count = n_patients - low_risk_count - moderate_risk_count
        
        # Assign base risk scores by quartile
        for i in range(n_patients):
            quartile = i // (n_patients // 4) if n_patients >= 4 else 0
            if quartile == 0:  # Lowest risk
                risk_scores[i] = np.random.uniform(0.0, 0.25)
            elif quartile == 1:  # Low-moderate risk
                risk_scores[i] = np.random.uniform(0.2, 0.45)
            elif quartile == 2:  # Moderate-high risk
                risk_scores[i] = np.random.uniform(0.4, 0.7)
            else:  # High risk
                risk_scores[i] = np.random.uniform(0.6, 1.0)
        
        # Add patient-specific factors
        risk_scores += 0.1 * (df['admission_type'] == 'EMERGENCY').astype(int)
        risk_scores += 0.05 * (df['insurance'] == 'Medicaid').astype(int)
        risk_scores += np.random.normal(0, 0.05, len(df))
        
        # Ensure scores are in valid range
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Convert to binary with thresholds that create varied distribution
        return (risk_scores > 0.5).astype(int)
    
    def _generate_readmission_risk(self, df):
        """Generate 30-day readmission risk with varied profiles"""
        n_patients = len(df)
        risk_scores = np.zeros(n_patients)
        
        # Create distinct risk distributions
        for i in range(n_patients):
            quartile = i // (n_patients // 4) if n_patients >= 4 else 0
            if quartile == 0:  # Lowest risk
                risk_scores[i] = np.random.uniform(0.0, 0.2)
            elif quartile == 1:  # Low-moderate risk
                risk_scores[i] = np.random.uniform(0.15, 0.35)
            elif quartile == 2:  # Moderate-high risk
                risk_scores[i] = np.random.uniform(0.3, 0.6)
            else:  # High risk
                risk_scores[i] = np.random.uniform(0.5, 0.9)
        
        # Add demographic risk factors
        risk_scores += 0.15 * (df['ethnicity'] == 'BLACK/AFRICAN AMERICAN').astype(int)
        risk_scores += 0.1 * (df['marital_status'] == 'SINGLE').astype(int)
        risk_scores += 0.08 * (df['insurance'] == 'Medicaid').astype(int)
        risk_scores += np.random.normal(0, 0.08, len(df))
        
        # Ensure scores are in valid range
        risk_scores = np.clip(risk_scores, 0, 1)
        
        # Convert to binary outcome
        return (risk_scores > 0.4).astype(int)
    
    def _generate_los(self, df):
        """Generate length of stay with risk correlation"""
        n_patients = len(df)
        base_los = np.zeros(n_patients)
        
        # Create varied LOS based on risk quartiles
        for i in range(n_patients):
            quartile = i // (n_patients // 4) if n_patients >= 4 else 0
            if quartile == 0:  # Low risk - shorter stays
                base_los[i] = np.random.exponential(2)  # Mean ~2 days
            elif quartile == 1:  # Low-moderate risk
                base_los[i] = np.random.exponential(3)  # Mean ~3 days
            elif quartile == 2:  # Moderate-high risk
                base_los[i] = np.random.exponential(4)  # Mean ~4 days
            else:  # High risk - longer stays
                base_los[i] = np.random.exponential(6)  # Mean ~6 days
        
        # Increase LOS for high-risk patients
        critical_multiplier = 1 + 1.5 * (df['mortality_30day'] == 1).astype(int)
        return base_los * critical_multiplier
    
    def _generate_conditions(self, df):
        """Generate patient conditions with varied risk profiles"""
        conditions_list = [
            'Hypertension', 'Diabetes', 'Heart Disease', 'COPD', 'Asthma',
            'Depression', 'Anxiety', 'Obesity', 'Hyperlipidemia', 'Arthritis',
            'Cancer', 'Stroke', 'Myocardial Infarction', 'Congestive Heart Failure', 'Chronic Kidney Disease'
        ]
        
        # Risk profile distribution: 40% low, 35% moderate, 25% high risk
        risk_profiles = []
        n_patients = len(df)
        # Low risk: 0-1 conditions
        low_risk_count = int(n_patients * 0.4)
        # Moderate risk: 2-3 conditions
        moderate_risk_count = int(n_patients * 0.35)
        # High risk: 4+ conditions
        high_risk_count = n_patients - low_risk_count - moderate_risk_count
        
        risk_profiles.extend([0] * low_risk_count)  # 0-1 conditions
        risk_profiles.extend([1] * moderate_risk_count)  # 2-3 conditions
        risk_profiles.extend([2] * high_risk_count)  # 4+ conditions
        
        # Shuffle to distribute risk profiles randomly
        np.random.shuffle(risk_profiles)
        
        records = []
        for idx, patient in df.iterrows():
            risk_level = risk_profiles[idx]
            
            if risk_level == 0:  # Low risk
                n_conditions = np.random.choice([0, 1], p=[0.6, 0.4])
                available_conditions = ['Asthma', 'Anxiety', 'Arthritis', 'Depression']
            elif risk_level == 1:  # Moderate risk
                n_conditions = np.random.choice([2, 3], p=[0.6, 0.4])
                available_conditions = ['Hypertension', 'Diabetes', 'Hyperlipidemia', 'COPD', 'Obesity']
            else:  # High risk
                n_conditions = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])
                available_conditions = conditions_list  # All conditions including high-risk ones
            
            if n_conditions > 0:
                patient_conditions = np.random.choice(available_conditions, 
                                                    size=min(n_conditions, len(available_conditions)), 
                                                    replace=False)
                
                for condition in patient_conditions:
                    records.append({
                        'patient_id': patient['patient_id'],
                        'condition': condition,
                        'diagnosis_date': patient['birth_date'] + timedelta(days=np.random.randint(365*20, 365*60)),
                        'status': np.random.choice(['ACTIVE', 'RESOLVED', 'INACTIVE'], p=[0.7, 0.2, 0.1])
                    })
        
        return pd.DataFrame(records)
    
    def _generate_medications(self, df):
        """Generate patient medications with varied risk profiles"""
        medications_list = [
            'Lisinopril', 'Metformin', 'Atorvastatin', 'Albuterol', 'Sertraline',
            'Aspirin', 'Levothyroxine', 'Insulin', 'Hydrochlorothiazide', 'Amlodipine',
            'Metoprolol', 'Warfarin', 'Furosemide', 'Digoxin', 'Clopidogrel'
        ]
        
        # Match medication count to condition risk profiles
        records = []
        for idx, patient in df.iterrows():
            # Get patient's conditions to determine medication needs
            # For simplicity, we'll create a correlation between patient index and medication count
            patient_risk_quartile = idx // (len(df) // 4) if len(df) >= 4 else 0
            
            if patient_risk_quartile == 0:  # Lowest quartile - 0-1 medications
                n_medications = np.random.choice([0, 1], p=[0.7, 0.3])
            elif patient_risk_quartile == 1:  # Low-moderate - 1-2 medications
                n_medications = np.random.choice([1, 2], p=[0.6, 0.4])
            elif patient_risk_quartile == 2:  # Moderate-high - 2-4 medications
                n_medications = np.random.choice([2, 3, 4], p=[0.4, 0.4, 0.2])
            else:  # Highest quartile - 3-6 medications
                n_medications = np.random.choice([3, 4, 5, 6], p=[0.3, 0.3, 0.2, 0.2])
            
            if n_medications > 0:
                patient_meds = np.random.choice(medications_list, 
                                              size=min(n_medications, len(medications_list)), 
                                              replace=False)
                
                for med in patient_meds:
                    records.append({
                        'patient_id': patient['patient_id'],
                        'medication': med,
                        'start_date': patient['birth_date'] + timedelta(days=np.random.randint(365*25, 365*55)),
                        'end_date': None if np.random.random() > 0.3 else 
                                   patient['birth_date'] + timedelta(days=np.random.randint(365*30, 365*60)),
                        'dosage': f"{np.random.choice([5, 10, 20, 40, 80])}mg"
                    })
        
        return pd.DataFrame(records)
    
    def _generate_procedures(self, df):
        """Generate patient procedures"""
        procedures_list = [
            'Chest X-Ray', 'Blood Test', 'EKG', 'MRI', 'CT Scan',
            'Colonoscopy', 'Mammogram', 'Physical Therapy', 'Vaccination', 'Surgery'
        ]
        
        records = []
        for _, patient in df.iterrows():
            n_procedures = np.random.poisson(3)  # Average 3 procedures per patient
            patient_procedures = np.random.choice(procedures_list, 
                                                size=min(n_procedures, len(procedures_list)), 
                                                replace=False)
            
            for proc in patient_procedures:
                records.append({
                    'patient_id': patient['patient_id'],
                    'procedure': proc,
                    'procedure_date': patient['birth_date'] + timedelta(days=np.random.randint(365*20, 365*60)),
                    'performing_physician': f"Dr. {np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])}",
                    'result': np.random.choice(['NORMAL', 'ABNORMAL', 'PENDING'])
                })
        
        return pd.DataFrame(records)
    
    def _generate_examination_data(self, df):
        """Generate NHANES-style examination data with varied risk profiles"""
        exam_data = []
        n_patients = len(df)
        
        # Create risk profile distribution for examination data
        for idx, patient in df.iterrows():
            # Risk quartiles: 0=low, 1=low-moderate, 2=moderate-high, 3=high
            risk_quartile = idx // (n_patients // 4) if n_patients >= 4 else 0
            
            if risk_quartile == 0:  # Low risk patients
                bp_sys = np.random.normal(120, 10)  # Normal BP
                bp_dia = np.random.normal(75, 8)
                bmi = np.random.normal(23, 3)  # Normal weight
            elif risk_quartile == 1:  # Low-moderate risk
                bp_sys = np.random.normal(135, 12)  # Prehypertension
                bp_dia = np.random.normal(85, 10)
                bmi = np.random.normal(27, 4)  # Overweight
            elif risk_quartile == 2:  # Moderate-high risk
                bp_sys = np.random.normal(150, 15)  # Stage 1 hypertension
                bp_dia = np.random.normal(95, 12)
                bmi = np.random.normal(32, 5)  # Obese
            else:  # High risk patients
                bp_sys = np.random.normal(170, 20)  # Stage 2 hypertension
                bp_dia = np.random.normal(105, 15)
                bmi = np.random.normal(38, 6)  # Severely obese
            
            # Ensure realistic ranges
            bp_sys = max(90, min(220, bp_sys))
            bp_dia = max(60, min(140, bp_dia))
            bmi = max(15, min(50, bmi))
            
            height = np.random.normal(170, 10) if patient['gender'] == 1 else np.random.normal(160, 8)
            height = max(140, min(200, height))
            weight = bmi * (height/100)**2
            
            exam_data.append({
                'seqn': patient['seqn'],
                'bp_systolic': bp_sys,
                'bp_diastolic': bp_dia,
                'height': height,
                'weight': weight,
                'bmi': bmi
            })
        return pd.DataFrame(exam_data)
    
    def _generate_laboratory_data(self, df):
        """Generate NHANES-style laboratory data with varied risk profiles"""
        lab_data = []
        n_patients = len(df)
        
        for idx, patient in df.iterrows():
            # Risk quartiles: match examination data
            risk_quartile = idx // (n_patients // 4) if n_patients >= 4 else 0
            
            if risk_quartile == 0:  # Low risk
                glucose = np.random.normal(90, 8)
                chol_total = np.random.normal(180, 25)
                hdl = np.random.normal(55, 10)
                ldl = np.random.normal(100, 20)
                triglycerides = np.random.normal(120, 30)
            elif risk_quartile == 1:  # Low-moderate risk
                glucose = np.random.normal(105, 12)
                chol_total = np.random.normal(210, 30)
                hdl = np.random.normal(45, 12)
                ldl = np.random.normal(130, 25)
                triglycerides = np.random.normal(160, 35)
            elif risk_quartile == 2:  # Moderate-high risk
                glucose = np.random.normal(125, 15)
                chol_total = np.random.normal(240, 35)
                hdl = np.random.normal(35, 15)
                ldl = np.random.normal(160, 30)
                triglycerides = np.random.normal(200, 40)
            else:  # High risk
                glucose = np.random.normal(150, 20)
                chol_total = np.random.normal(280, 45)
                hdl = np.random.normal(30, 12)
                ldl = np.random.normal(190, 35)
                triglycerides = np.random.normal(300, 60)
            
            # Ensure realistic clinical ranges
            glucose = max(70, min(300, glucose))
            chol_total = max(120, min(400, chol_total))
            hdl = max(20, min(100, hdl))
            ldl = max(50, min(300, ldl))
            triglycerides = max(50, min(600, triglycerides))
            
            lab_data.append({
                'seqn': patient['seqn'],
                'glucose': glucose,
                'cholesterol_total': chol_total,
                'hdl_cholesterol': hdl,
                'ldl_cholesterol': ldl,
                'triglycerides': triglycerides,
                'hba1c': 5.0 + (glucose - 90) / 20,  # Correlated with glucose
                'creatinine': np.random.normal(1.0, 0.2),
                'bun': np.random.normal(15, 5)
            })
        return pd.DataFrame(lab_data)

# Example usage
if __name__ == "__main__":
    loader = HealthcareDataLoader()
    
    print("Healthcare Data Loader - Available Datasets:")
    for key, desc in loader.datasets.items():
        print(f"  • {key}: {desc}")
    
    print("\n" + "="*50)
    
    # Load sample data
    mimic_data = loader.load_mimic_sample(500)
    print(f"\nMIMIC Sample - Shape: {mimic_data.shape}")
    print("Columns:", list(mimic_data.columns)[:10], "...")
    
    print("\n" + "="*50)
    
    synthea_data = loader.load_synthea_data(300)
    print(f"\nSynthea Sample - Patient records: {synthea_data[0].shape}")
    print(f"Conditions: {synthea_data[1].shape}")
    print(f"Medications: {synthea_data[2].shape}")
    print(f"Procedures: {synthea_data[3].shape}")
