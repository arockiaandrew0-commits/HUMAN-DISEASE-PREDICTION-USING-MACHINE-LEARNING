import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, abort
import pandas as pd

import logging
import logging.config
import time
import uuid
from collections import defaultdict
from flask_cors import CORS

# data/, model/ and logs/ are resolved relative to this file, so the API can be
# started from any working directory (e.g. `python backend/app.py`).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# The UI lives in ../frontend. It can be hosted separately, but the backend also
# serves it so a single `python backend/app.py` gives a working app on one URL.
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(__name__, static_folder=None)

# Initialize tracking variables for metrics
START_TIME = time.time()
REQUEST_COUNTERS = defaultdict(int)
ENDPOINT_LATENCIES = defaultdict(list)

# Symptom causes information - expanded to cover all symptoms
symptom_causes = {
    "headache": "Headaches can be caused by stress, dehydration, eye strain, sinus congestion, or more serious conditions like migraines or high blood pressure. They typically affect the forehead, temples, and back of the neck.",
    "fever": "Fever is your body's way of fighting infection. It's commonly caused by viral or bacterial infections, inflammatory conditions, or as a response to certain medications.",
    "high_fever": "High fever (typically above 39°C or 102°F) can indicate serious infections, inflammatory conditions, or heat-related illness. Seek medical attention if it persists or is accompanied by severe symptoms.",
    "cough": "Coughing is a reflex that clears your throat and airways. It can be caused by infections, allergies, asthma, acid reflux, or irritants in the air.",
    "sore_throat": "A sore throat can result from viral infections like the common cold, bacterial infections like strep throat, allergies, dry air, or irritants like smoke.",
    "fatigue": "Fatigue can be caused by lack of sleep, poor diet, stress, depression, anemia, or underlying medical conditions like thyroid disorders.",
    "body_ache": "Body aches often occur with infections as your immune system fights back. They can also be caused by inflammation, stress, or overexertion.",
    "body_pain": "Body pain often occurs with infections as your immune system fights back. They can also be caused by inflammation, stress, or overexertion.",
    "shortness_of_breath": "Difficulty breathing can be caused by anxiety, asthma, allergies, respiratory infections, heart problems, or lung conditions.",
    "shortness_breath": "Difficulty breathing can be caused by anxiety, asthma, allergies, respiratory infections, heart problems, or lung conditions.",
    "chest_pain": "Chest pain may be caused by heart problems, lung conditions, digestive issues, muscle strain, or anxiety. Always take chest pain seriously.",
    "dizziness": "Dizziness can result from inner ear problems, low blood pressure, dehydration, anemia, or medication side effects.",
    "blurred_vision": "Blurred vision may be caused by eye strain, need for glasses, eye infections, cataracts, or more serious conditions affecting the eyes or brain.",
    "frequent_urination": "Frequent urination can be caused by urinary tract infections, diabetes, pregnancy, enlarged prostate, or drinking too much fluid.",
    "excessive_thirst": "Excessive thirst often accompanies diabetes, but can also be caused by medications, loss of blood, severe burns, or severe vomiting or diarrhea.",
    "wheezing": "Wheezing is often associated with asthma, allergies, respiratory infections, or chronic obstructive pulmonary disease (COPD).",
    "runny_nose": "A runny nose is typically caused by allergies, the common cold, flu, or sinusitis. It's your body's way of flushing out irritants.",
    "sneezing": "Sneezing is typically caused by allergies, irritants in the air, or respiratory infections like the common cold or flu.",
    "nausea": "Nausea can be caused by food poisoning, motion sickness, pregnancy, migraines, medication side effects, or digestive disorders.",
    "vomiting": "Vomiting can result from food poisoning, viral infections, migraines, pregnancy, motion sickness, or as a response to certain medications.",
    "diarrhea": "Diarrhea is often caused by viral or bacterial infections, food intolerances, medications, or digestive disorders like IBS.",
    "stomach_pain": "Stomach pain can be caused by indigestion, gas, constipation, food poisoning, ulcers, or inflammatory bowel conditions.",
    "abdominal_pain": "Abdominal pain can be caused by indigestion, gas, constipation, food poisoning, ulcers, or inflammatory bowel conditions.",
    "rash": "Rashes can be caused by allergies, infections, contact with irritants, autoimmune conditions, or reactions to medications.",
    "itchy_rash": "Itchy rashes can be caused by allergies, contact dermatitis, infections, or skin conditions like eczema or psoriasis.",
    "joint_pain": "Joint pain can be caused by arthritis, injuries, infections, autoimmune conditions, or overuse.",
    "muscle_pain": "Muscle pain can be caused by overuse, tension, infections, or conditions like fibromyalgia.",
    "chills": "Chills often accompany fever and can be caused by infections, exposure to cold, or certain medical conditions.",
    "night_sweats": "Night sweats can be caused by infections, hormonal changes, medications, or underlying medical conditions.",
    "weight_loss": "Unexplained weight loss can be caused by metabolic disorders, digestive issues, chronic infections, or serious underlying conditions.",
    "confusion": "Confusion can be caused by infections, dehydration, medication side effects, or neurological conditions. Seek immediate medical attention if severe.",
    "pale_skin": "Pale skin can be caused by anemia, poor circulation, shock, or underlying medical conditions.",
    "jaundice": "Jaundice (yellowing of skin and eyes) can be caused by liver problems, bile duct obstruction, or blood disorders. Requires medical evaluation.",
    "migraine_like": "Migraine-like symptoms can include severe headaches, sensitivity to light and sound, and nausea. Can be triggered by stress, hormonal changes, or certain foods.",
    "loss_smell": "Loss of smell can be caused by nasal congestion, infections (like COVID-19), head injuries, or neurological conditions.",
    "loss_taste": "Loss of taste can be caused by infections (like COVID-19), head injuries, neurological conditions, or certain medications.",
    "dry_cough": "A dry cough (non-productive) can be caused by allergies, asthma, viral infections, or environmental irritants. It doesn't produce phlegm.",
    "wet_cough": "A wet or productive cough produces phlegm and is often caused by infections, bronchitis, pneumonia, or conditions affecting the lungs.",
    "nasal_congestion": "Nasal congestion can be caused by allergies, colds, sinus infections, or structural issues with the nasal passages.",
    "back_pain": "Back pain can be caused by muscle strain, poor posture, injuries, herniated discs, arthritis, or underlying medical conditions.",
    "neck_pain": "Neck pain can result from muscle strain, poor posture, injuries, arthritis, or conditions affecting the cervical spine.",
    "ear_pain": "Ear pain can be caused by ear infections, sinus infections, changes in pressure, temporomandibular joint (TMJ) disorders, or referred pain.",
    "sinus_pressure": "Sinus pressure can be caused by sinus infections, allergies, colds, or structural issues with the sinuses.",
    "heartburn": "Heartburn is caused by stomach acid refluxing into the esophagus. It can be triggered by certain foods, medications, or underlying conditions like GERD.",
    "indigestion": "Indigestion can be caused by eating too quickly, spicy or fatty foods, stress, medications, or underlying digestive disorders.",
    "constipation": "Constipation can be caused by lack of fiber, dehydration, lack of exercise, medications, or underlying digestive conditions.",
    "bloating": "Bloating can be caused by gas, overeating, food intolerances, digestive disorders, or hormonal changes.",
    "gas": "Excessive gas can be caused by swallowing air, eating certain foods, digestive disorders, or bacterial fermentation in the gut.",
    "difficulty_swallowing": "Difficulty swallowing can be caused by throat infections, muscle problems, nerve disorders, or structural issues. Seek medical attention if severe.",
    "insomnia": "Insomnia can be caused by stress, anxiety, medications, medical conditions, or lifestyle factors affecting sleep.",
    "anxiety": "Anxiety can be a normal response to stress, or it may indicate an anxiety disorder. It can be triggered by various factors including stress, medical conditions, or genetics.",
    "irritability": "Irritability can be caused by stress, lack of sleep, hormonal changes, mental health conditions, or underlying medical issues.",
    "memory_problems": "Memory problems can be caused by stress, lack of sleep, medications, aging, or more serious conditions affecting the brain.",
    "sensitivity_to_light": "Light sensitivity can be caused by migraines, eye conditions, infections, or neurological conditions.",
    "sensitivity_to_sound": "Sound sensitivity can be caused by migraines, ear conditions, anxiety, or neurological conditions.",
    "ringing_in_ears": "Ringing in ears (tinnitus) can be caused by exposure to loud noise, ear infections, medications, or underlying medical conditions.",
    "numbness": "Numbness can be caused by nerve compression, injuries, diabetes, neurological conditions, or poor circulation.",
    "tingling": "Tingling sensations can be caused by nerve compression, poor circulation, diabetes, or neurological conditions.",
    "weakness": "Weakness can be caused by fatigue, infections, anemia, neurological conditions, or muscle disorders.",
    "swelling": "Swelling can be caused by injuries, infections, inflammation, fluid retention, or underlying medical conditions.",
    "joint_swelling": "Joint swelling can be caused by arthritis, injuries, infections, or inflammatory conditions affecting the joints.",
    "itching": "Itching can be caused by dry skin, allergies, skin conditions, medications, or underlying medical conditions.",
    "dry_skin": "Dry skin can be caused by environmental factors, dehydration, certain medications, or skin conditions.",
    "hair_loss": "Hair loss can be caused by genetics, hormonal changes, stress, medications, medical conditions, or nutritional deficiencies.",
    "dry_mouth": "Dry mouth can be caused by dehydration, medications, medical conditions like Sjögren's syndrome, or breathing through the mouth.",
    "bad_breath": "Bad breath can be caused by poor oral hygiene, dental problems, infections, dry mouth, or certain foods.",
    "mouth_sores": "Mouth sores can be caused by viral infections, injuries, autoimmune conditions, or nutritional deficiencies.",
    "swollen_lymph_nodes": "Swollen lymph nodes can be caused by infections, immune system responses, or more serious conditions. Seek medical evaluation if persistent.",
    "bruising": "Easy bruising can be caused by medications, aging, blood disorders, or underlying medical conditions.",
    "bleeding": "Excessive bleeding can be caused by injuries, blood disorders, medications, or underlying medical conditions. Seek immediate care if severe.",
    "cold_hands_feet": "Cold hands and feet can be caused by poor circulation, Raynaud's phenomenon, anemia, or thyroid conditions.",
    "hot_flashes": "Hot flashes can be caused by hormonal changes, menopause, medications, or certain medical conditions.",
    "irregular_heartbeat": "Irregular heartbeat (arrhythmia) can be caused by stress, caffeine, medications, heart conditions, or electrolyte imbalances. Seek medical attention if severe.",
    "palpitations": "Palpitations (feeling of rapid or irregular heartbeat) can be caused by stress, caffeine, medications, anxiety, or heart conditions.",
    "frequent_urination": "Frequent urination can be caused by urinary tract infections, diabetes, pregnancy, enlarged prostate, or drinking too much fluid.",
    "burning_urination": "Burning urination can be caused by urinary tract infections, irritation, sexually transmitted infections, or other conditions affecting the urinary tract.",
    "blood_in_urine": "Blood in urine can be caused by infections, kidney stones, injuries, or more serious conditions. Requires medical evaluation.",
    "difficulty_concentrating": "Difficulty concentrating can be caused by stress, lack of sleep, ADHD, anxiety, depression, or underlying medical conditions.",
    "mood_swings": "Mood swings can be caused by hormonal changes, stress, mental health conditions, medications, or underlying medical issues."
}

# Symptom name normalization mapping - maps variations to standard names
SYMPTOM_NAME_MAP = {
    "shortness_breath": "shortness_of_breath",
    "body_pain": "body_ache",
    "abdominal_pain": "stomach_pain"
}

def normalize_symptom_name(symptom):
    """Normalize symptom name to handle variations"""
    if not symptom:
        return None
    symptom_lower = symptom.lower().strip()
    
    # First check direct mapping for known variations
    if symptom_lower in SYMPTOM_NAME_MAP:
        return SYMPTOM_NAME_MAP[symptom_lower]
    
    # Then check if it exists directly in symptom_causes
    if symptom_lower in symptom_causes:
        return symptom_lower
    
    # Try variations (replace underscores with spaces and vice versa)
    # Remove 'symptom_' prefix if present
    if symptom_lower.startswith('symptom_'):
        symptom_lower = symptom_lower[8:]
    
    variations = [
        symptom_lower,
        symptom_lower.replace('_', ' '),
        symptom_lower.replace(' ', '_'),
        symptom_lower.replace('-', '_'),
        symptom_lower.replace('-', ' ')
    ]
    
    for var in variations:
        if var in symptom_causes:
            return var
    
    # Return the normalized version (with underscores) for consistency
    return symptom_lower.replace(' ', '_').replace('-', '_')

# Simple disease prediction model using decision rules
class SimpleDiseasePredictionModel:
    def __init__(self):
        self.diseases = ['Healthy', 'Flu', 'Pneumonia', 'Hypertension', 'Diabetes']
        
    def predict(self, features):
        """
        Predict disease based on simple rules
        features: [age, temperature, heart_rate, blood_pressure, oxygen_level, pain_level]
        """
        age, temperature, heart_rate, blood_pressure, oxygen_level, pain_level = features[0]
        
        # Simple rule-based prediction
        if temperature > 38.5 and heart_rate > 90 and oxygen_level < 95:
            return ['Pneumonia']
        elif temperature > 38 and heart_rate > 85:
            return ['Flu']
        elif blood_pressure > 140:
            return ['Hypertension']
        elif age > 50 and blood_pressure > 130:
            return ['Diabetes']
        else:
            return ['Healthy']
    
    def predict_proba(self, features):
        """Return confidence scores for the prediction"""
        prediction = self.predict(features)[0]
        confidences = [0.0] * len(self.diseases)
        
        # Set confidence for the predicted disease
        disease_index = self.diseases.index(prediction)
        
        # Simple confidence calculation
        if prediction == 'Pneumonia':
            confidences[disease_index] = 0.85
        elif prediction == 'Flu':
            confidences[disease_index] = 0.75
        elif prediction == 'Hypertension':
            confidences[disease_index] = 0.80
        elif prediction == 'Diabetes':
            confidences[disease_index] = 0.70
        else:  # Healthy
            confidences[disease_index] = 0.65
            
        return [confidences]

# Symptom-based disease prediction model
class SymptomBasedDiseaseModel:
    def __init__(self):
        self.diseases = ['Healthy', 'Flu', 'Pneumonia', 'Hypertension', 'Diabetes', 
                         'Common Cold', 'Bronchitis', 'Asthma', 'Migraine', 'Food Poisoning']
        
        # Mapping of symptoms to diseases with weights
        self.symptom_disease_map = {
            'fever': {'Flu': 0.7, 'Pneumonia': 0.6, 'Common Cold': 0.5},
            'high_fever': {'Pneumonia': 0.7, 'Flu': 0.6},
            'cough': {'Flu': 0.6, 'Pneumonia': 0.7, 'Common Cold': 0.6, 'Bronchitis': 0.8, 'Asthma': 0.5},
            'dry_cough': {'Asthma': 0.7, 'Common Cold': 0.5, 'Bronchitis': 0.6},
            'wet_cough': {'Bronchitis': 0.8, 'Pneumonia': 0.7, 'Flu': 0.5},
            'headache': {'Flu': 0.5, 'Common Cold': 0.4, 'Migraine': 0.9},
            'sore_throat': {'Flu': 0.6, 'Common Cold': 0.7},
            'fatigue': {'Flu': 0.5, 'Pneumonia': 0.5, 'Hypertension': 0.3, 'Diabetes': 0.4},
            'body_ache': {'Flu': 0.7, 'Common Cold': 0.3},
            'body_pain': {'Flu': 0.7, 'Common Cold': 0.3},
            'shortness_of_breath': {'Pneumonia': 0.8, 'Asthma': 0.9},
            'shortness_breath': {'Pneumonia': 0.8, 'Asthma': 0.9},
            'chest_pain': {'Pneumonia': 0.6, 'Hypertension': 0.5, 'Asthma': 0.6},
            'dizziness': {'Hypertension': 0.7, 'Migraine': 0.6},
            'blurred_vision': {'Hypertension': 0.5, 'Diabetes': 0.5, 'Migraine': 0.7},
            'frequent_urination': {'Diabetes': 0.8},
            'excessive_thirst': {'Diabetes': 0.8},
            'wheezing': {'Asthma': 0.9, 'Bronchitis': 0.7},
            'runny_nose': {'Common Cold': 0.8, 'Flu': 0.4},
            'sneezing': {'Common Cold': 0.7, 'Flu': 0.3},
            'nasal_congestion': {'Common Cold': 0.8, 'Flu': 0.4},
            'nausea': {'Food Poisoning': 0.8, 'Migraine': 0.6},
            'vomiting': {'Food Poisoning': 0.9},
            'diarrhea': {'Food Poisoning': 0.9},
            'stomach_pain': {'Food Poisoning': 0.7},
            'abdominal_pain': {'Food Poisoning': 0.7},
            'heartburn': {'Food Poisoning': 0.3},
            'indigestion': {'Food Poisoning': 0.4},
            'constipation': {'Food Poisoning': 0.2},
            'back_pain': {'Flu': 0.3},
            'ear_pain': {'Common Cold': 0.5, 'Flu': 0.3},
            'sinus_pressure': {'Common Cold': 0.6, 'Flu': 0.4},
            'joint_pain': {'Flu': 0.4},
            'muscle_pain': {'Flu': 0.6},
            'chills': {'Flu': 0.6, 'Pneumonia': 0.5},
            'migraine_like': {'Migraine': 0.9},
            'sensitivity_to_light': {'Migraine': 0.8},
            'sensitivity_to_sound': {'Migraine': 0.8},
            'loss_smell': {'Common Cold': 0.4},
            'loss_taste': {'Common Cold': 0.4},
            'insomnia': {'Hypertension': 0.3},
            'anxiety': {'Hypertension': 0.4},
            'palpitations': {'Hypertension': 0.6},
            'irregular_heartbeat': {'Hypertension': 0.7},
            'burning_urination': {'Food Poisoning': 0.3}
        }
    
    def predict(self, symptoms):
        """
        Predict disease based on symptoms
        symptoms: list of symptom strings
        """
        if not symptoms:
            return ['Healthy']
        
        # Calculate scores for each disease based on symptoms
        disease_scores = {disease: 0.0 for disease in self.diseases}
        
        for symptom in symptoms:
            if symptom in self.symptom_disease_map:
                for disease, weight in self.symptom_disease_map[symptom].items():
                    disease_scores[disease] += weight
        
        # Find the disease with the highest score
        max_score = 0
        predicted_disease = 'Healthy'
        
        for disease, score in disease_scores.items():
            if score > max_score:
                max_score = score
                predicted_disease = disease
        
        # If no significant score, return Healthy
        if max_score < 0.5:
            return ['Healthy']
        
        return [predicted_disease]
    
    def predict_proba(self, symptoms):
        """Return confidence scores for the prediction"""
        if not symptoms:
            confidences = [0.0] * len(self.diseases)
            healthy_index = self.diseases.index('Healthy')
            confidences[healthy_index] = 0.6
            return [confidences]
        
        # Calculate scores for each disease based on symptoms
        disease_scores = {disease: 0.0 for disease in self.diseases}
        
        for symptom in symptoms:
            if symptom in self.symptom_disease_map:
                for disease, weight in self.symptom_disease_map[symptom].items():
                    disease_scores[disease] += weight
        
        # Convert scores to probabilities
        total_score = sum(disease_scores.values())
        if total_score == 0:
            confidences = [0.0] * len(self.diseases)
            healthy_index = self.diseases.index('Healthy')
            confidences[healthy_index] = 0.6
            return [confidences]
        
        confidences = [0.0] * len(self.diseases)
        for i, disease in enumerate(self.diseases):
            if disease in disease_scores:
                # Normalize and apply a sigmoid-like function to get probabilities
                confidences[i] = min(0.95, disease_scores[disease] / (total_score + 1))
        
        return [confidences]

# Create model instances
model = SimpleDiseasePredictionModel()
symptom_model = SymptomBasedDiseaseModel()

@app.route('/')
def home():
    """Serve the web UI from ../frontend."""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def frontend_asset(filename):
    """Serve frontend files (css/, js/). API routes take precedence over this."""
    if not os.path.isfile(os.path.join(FRONTEND_DIR, filename)):
        abort(404)
    return send_from_directory(FRONTEND_DIR, filename)

@app.route('/api')
def api_index():
    """API index: lists available endpoints."""
    return jsonify({
        'name': 'MedPredict Disease Prediction API',
        'status': 'ok',
        'endpoints': {
            'GET /health': 'Service and model status',
            'GET /symptoms': 'List of supported symptoms',
            'GET /symptom_info?symptom=<name>': 'Causes, prevalence, red flags and advice for a symptom',
            'POST /predict_by_symptoms_ml': 'ML prediction from {"symptoms": [...]}',
            'POST /predict': 'Rule-based vitals screen',
            'POST /chat': 'Free-text symptom assistant',
            'GET /metrics': 'Request counters and latencies',
        },
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = np.array([[
            float(data['age']),
            float(data['temperature']),
            float(data['heart_rate']),
            float(data['blood_pressure']),
            float(data['oxygen_level']),
            float(data['pain_level'])
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Get the highest probability
        max_proba_index = model.diseases.index(prediction)
        max_proba = prediction_proba[max_proba_index] * 100
        
        return jsonify({
            'disease': prediction,
            'confidence': round(float(max_proba), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Cached dataset and symptom stats
SYMPTOM_DF = None
SYMPTOM_STATS_CACHE = {}

def _get_dataset_path():
    pro = os.path.join(BASE_DIR, 'data', 'symptom_dataset_pro.csv')
    base = os.path.join(BASE_DIR, 'data', 'symptom_dataset.csv')
    if os.path.exists(pro):
        return pro
    if os.path.exists(base):
        return base
    return None

def _load_symptom_df():
    global SYMPTOM_DF
    if SYMPTOM_DF is None:
        path = _get_dataset_path()
        if path:
            try:
                SYMPTOM_DF = pd.read_csv(path)
            except Exception:
                SYMPTOM_DF = None
    return SYMPTOM_DF

RED_FLAGS_MAP = {
    'chest_pain': [
        'Severe or pressure-like pain',
        'Pain with shortness of breath or sweating',
        'Pain radiating to arm or jaw'
    ],
    'shortness_of_breath': [
        'Sudden or severe breathing difficulty',
        'Shortness of breath at rest',
        'Associated chest pain or fainting'
    ],
    'shortness_breath': [
        'Sudden or severe breathing difficulty',
        'Shortness of breath at rest',
        'Associated chest pain or fainting'
    ],
    'blurred_vision': [
        'Sudden vision loss in one or both eyes',
        'Vision changes with severe headache or weakness'
    ],
    'stomach_pain': [
        'Severe pain with rigid abdomen',
        'Blood in vomit or stool'
    ],
    'abdominal_pain': [
        'Severe pain with rigid abdomen',
        'Blood in vomit or stool'
    ],
    'fever': [
        'Fever > 39.4°C (103°F) for >3 days',
        'Fever with stiff neck, confusion, or rash'
    ],
    'high_fever': [
        'Fever > 39.4°C (103°F) for >3 days',
        'Fever with stiff neck, confusion, or rash'
    ],
    'confusion': [
        'Sudden onset',
        'Associated with head injury',
        'Severe or worsening confusion'
    ],
    'jaundice': [
        'Yellowing of eyes or skin',
        'Dark urine or pale stools',
        'Requires immediate medical evaluation'
    ],
    'vomiting': [
        'Persistent vomiting',
        'Blood in vomit',
        'Unable to keep fluids down'
    ],
    'weight_loss': [
        'Unexplained significant weight loss',
        'Loss of appetite',
        'Associated with other concerning symptoms'
    ],
    'blood_in_urine': [
        'Any blood in urine requires immediate medical evaluation',
        'Associated with pain or fever',
        'Occurs after injury'
    ],
    'difficulty_swallowing': [
        'Complete inability to swallow',
        'Associated with drooling or choking',
        'Sudden onset'
    ],
    'irregular_heartbeat': [
        'Rapid or irregular heartbeat with chest pain',
        'Associated with fainting or dizziness',
        'Severe palpitations'
    ],
    'bleeding': [
        'Excessive or uncontrolled bleeding',
        "Bleeding that won't stop with pressure",
        'Associated with injury or head trauma'
    ]
}

HOME_ADVICE_MAP = {
    'headache': ['Rest in a quiet, dark room', 'Hydrate well', 'Consider OTC pain relief as directed'],
    'fever': ['Hydrate frequently', 'Wear light clothing', 'Consider acetaminophen/ibuprofen as directed'],
    'high_fever': ['Seek medical attention if above 39°C/102°F', 'Stay hydrated', 'Monitor closely'],
    'cough': ['Sip warm fluids', 'Honey for adults', 'Avoid smoke/irritants'],
    'sore_throat': ['Warm salt-water gargles', 'Stay hydrated', 'Use throat lozenges'],
    'fatigue': ['Prioritize sleep', 'Reduce stress', 'Balanced nutrition'],
    'body_ache': ['Rest', 'Warm baths or gentle stretching', 'Consider OTC pain relief as directed'],
    'body_pain': ['Rest', 'Warm baths or gentle stretching', 'Consider OTC pain relief as directed'],
    'shortness_of_breath': ['Rest and monitor', 'Use prescribed inhalers if applicable'],
    'shortness_breath': ['Rest and monitor', 'Use prescribed inhalers if applicable'],
    'chest_pain': ['Rest, avoid exertion', 'Seek urgent care if severe'],
    'dizziness': ['Move slowly, sit or lie down', 'Stay hydrated', 'Avoid sudden position changes'],
    'nausea': ['Small sips of clear fluids', 'BRAT-style diet as tolerated'],
    'vomiting': ['Small sips of clear fluids', 'Avoid solid foods until resolved', 'Seek care if persistent'],
    'diarrhea': ['Oral rehydration', 'Avoid dairy/fatty foods'],
    'stomach_pain': ['Small, bland meals', 'Avoid spicy or fatty foods', 'Monitor for worsening'],
    'abdominal_pain': ['Small, bland meals', 'Avoid spicy or fatty foods', 'Monitor for worsening'],
    'runny_nose': ['Use saline nasal sprays', 'Stay hydrated', 'Consider decongestants as directed'],
    'sneezing': ['Avoid allergens if possible', 'Wash hands frequently', 'Consider antihistamines if allergy-related'],
    'rash': ['Keep area clean and dry', 'Avoid scratching', 'Use mild moisturizers'],
    'itchy_rash': ['Keep area clean and dry', 'Avoid scratching', 'Consider antihistamines or topical treatments'],
    'joint_pain': ['Rest affected joints', 'Apply ice or heat', 'Gentle movement as tolerated'],
    'muscle_pain': ['Rest', 'Gentle stretching', 'Consider OTC pain relief as directed'],
    'chills': ['Stay warm', 'Hydrate well', 'Monitor temperature'],
    'night_sweats': ['Use breathable bedding', 'Keep room cool', 'Consider medical evaluation if persistent'],
    'weight_loss': ['Monitor nutrition', 'Keep a food diary', 'Consult healthcare provider'],
    'confusion': ['Seek immediate medical attention', 'Ensure safety', 'Monitor closely'],
    'pale_skin': ['Monitor for other symptoms', 'Check for anemia', 'Consult healthcare provider'],
    'jaundice': ['Seek immediate medical evaluation', 'Monitor for other symptoms', 'Avoid alcohol'],
    'migraine_like': ['Rest in dark, quiet room', 'Stay hydrated', 'Consider prescribed medications'],
    'loss_smell': ['Monitor for other symptoms', 'Consider COVID-19 testing', 'Consult healthcare provider'],
    'loss_taste': ['Monitor for other symptoms', 'Consider COVID-19 testing', 'Consult healthcare provider'],
    'excessive_thirst': ['Monitor blood sugar', 'Stay hydrated', 'Consult healthcare provider if persistent'],
    'wheezing': ['Use prescribed inhalers', 'Avoid triggers', 'Seek care if severe'],
    'blurred_vision': ['Rest eyes', 'Check if glasses prescription is current', 'Seek urgent care if sudden'],
    'dry_cough': ['Stay hydrated', 'Use humidifier', 'Avoid irritants like smoke'],
    'wet_cough': ['Stay hydrated', 'Expectorants may help', 'Monitor for worsening'],
    'nasal_congestion': ['Saline nasal sprays', 'Steam inhalation', 'Consider decongestants as directed'],
    'back_pain': ['Rest and gentle movement', 'Apply ice or heat', 'Maintain good posture'],
    'neck_pain': ['Gentle neck stretches', 'Apply heat or ice', 'Maintain proper posture'],
    'ear_pain': ['Apply warm compress', 'Avoid inserting objects', 'Consult if severe or with discharge'],
    'sinus_pressure': ['Steam inhalation', 'Saline nasal sprays', 'Stay hydrated'],
    'heartburn': ['Avoid trigger foods', 'Eat smaller meals', 'Avoid lying down after eating'],
    'indigestion': ['Eat slowly', 'Avoid large meals', 'Consider antacids as directed'],
    'constipation': ['Increase fiber intake', 'Stay hydrated', 'Regular exercise'],
    'bloating': ['Avoid gas-producing foods', 'Eat smaller meals', 'Gentle movement'],
    'gas': ['Avoid trigger foods', 'Eat slowly', 'Consider over-the-counter remedies'],
    'difficulty_swallowing': ['Eat soft foods', 'Take small bites', 'Seek medical attention if severe'],
    'insomnia': ['Maintain sleep schedule', 'Create relaxing bedtime routine', 'Limit screen time before bed'],
    'anxiety': ['Deep breathing exercises', 'Practice relaxation techniques', 'Consider professional help if persistent'],
    'irritability': ['Manage stress', 'Ensure adequate sleep', 'Consider underlying causes'],
    'memory_problems': ['Mental exercises', 'Stay organized with notes', 'Ensure adequate sleep'],
    'sensitivity_to_light': ['Wear sunglasses', 'Dim lights', 'Rest in dark room'],
    'sensitivity_to_sound': ['Reduce noise exposure', 'Use ear protection if needed', 'Create quiet environment'],
    'ringing_in_ears': ['Avoid loud noises', 'Background noise may help', 'Consult healthcare provider'],
    'numbness': ['Change position frequently', 'Gentle movement', 'Consult if persistent'],
    'tingling': ['Change position', 'Gentle movement', 'Monitor for other symptoms'],
    'weakness': ['Rest adequately', 'Ensure proper nutrition', 'Gradual activity as tolerated'],
    'swelling': ['Elevate affected area', 'Apply cold compress', 'Monitor for worsening'],
    'joint_swelling': ['Rest affected joint', 'Apply ice', 'Consider anti-inflammatory as directed'],
    'itching': ['Avoid scratching', 'Keep area moisturized', 'Cool compresses'],
    'dry_skin': ['Moisturize regularly', 'Avoid hot showers', 'Use gentle soaps'],
    'hair_loss': ['Gentle hair care', 'Balanced nutrition', 'Consult healthcare provider'],
    'dry_mouth': ['Stay hydrated', 'Sip water frequently', 'Avoid caffeine and alcohol'],
    'bad_breath': ['Maintain oral hygiene', 'Stay hydrated', 'Consider dental checkup'],
    'mouth_sores': ['Avoid spicy/acidic foods', 'Gentle oral hygiene', 'Monitor for healing'],
    'swollen_lymph_nodes': ['Monitor size and tenderness', 'Warm compress', 'Consult if persistent'],
    'bruising': ['Protect from injury', 'Apply ice if fresh', 'Consult if excessive'],
    'bleeding': ['Apply pressure', 'Elevate if possible', 'Seek immediate care if severe'],
    'cold_hands_feet': ['Keep warm', 'Gentle movement', 'Monitor for other symptoms'],
    'hot_flashes': ['Wear breathable clothing', 'Keep cool', 'Identify triggers'],
    'irregular_heartbeat': ['Avoid stimulants', 'Rest', 'Seek medical attention if severe'],
    'palpitations': ['Avoid caffeine', 'Practice relaxation', 'Monitor triggers'],
    'frequent_urination': ['Limit fluids before bed', 'Monitor intake', 'Consult if excessive'],
    'burning_urination': ['Increase water intake', 'Urinate frequently', 'Consult healthcare provider'],
    'blood_in_urine': ['Seek immediate medical evaluation', 'Note other symptoms', 'Avoid strenuous activity'],
    'difficulty_concentrating': ['Reduce distractions', 'Break tasks into smaller parts', 'Ensure adequate sleep'],
    'mood_swings': ['Maintain routine', 'Practice stress management', 'Consider professional help if severe'],
}

def compute_symptom_stats(symptom: str):
    if not symptom:
        return None
    
    # Normalize symptom name first
    normalized_symptom = normalize_symptom_name(symptom) or symptom
    
    # Use normalized name for cache key
    cache_key = normalized_symptom
    if cache_key in SYMPTOM_STATS_CACHE:
        return SYMPTOM_STATS_CACHE[cache_key]
    
    df = _load_symptom_df()
    if df is None:
        SYMPTOM_STATS_CACHE[cache_key] = {
            'dataset_source': None,
            'prevalence_pct': None,
            'top_diseases': [],
            'co_symptoms': [],
            'red_flags': RED_FLAGS_MAP.get(normalized_symptom, []),
            'advice': HOME_ADVICE_MAP.get(normalized_symptom, ['Monitor symptoms', 'Seek care if worsening'])
        }
        return SYMPTOM_STATS_CACHE[cache_key]
    
    # Try original symptom name first, then normalized
    col = f'symptom_{symptom}'
    if col not in df.columns:
        col = f'symptom_{normalized_symptom}'
    
    if col not in df.columns:
        SYMPTOM_STATS_CACHE[cache_key] = {
            'dataset_source': _get_dataset_path(),
            'prevalence_pct': None,
            'top_diseases': [],
            'co_symptoms': [],
            'red_flags': RED_FLAGS_MAP.get(normalized_symptom, []),
            'advice': HOME_ADVICE_MAP.get(normalized_symptom, ['Monitor symptoms', 'Seek care if worsening'])
        }
        return SYMPTOM_STATS_CACHE[cache_key]

    # Compute prevalence
    prevalence_pct = float(df[col].mean() * 100.0)

    # Filter rows where symptom present
    present = df[df[col] == 1]

    # Top diseases when symptom present (as percentage within present subset)
    top_diseases = []
    if not present.empty:
        d_counts = present['disease'].value_counts(normalize=True) * 100.0
        top_diseases = [
            {'name': str(d), 'percent': round(float(p), 2)}
            for d, p in d_counts.sort_values(ascending=False).head(5).items()
        ]

    # Co-occurring symptoms ranked by count in present subset
    co_symptoms = []
    symptom_cols = [c for c in df.columns if c.startswith('symptom_') and c != col]
    if symptom_cols and not present.empty:
        co_counts = present[symptom_cols].sum().sort_values(ascending=False)
        for name, val in co_counts.head(10).items():
            human = name.replace('symptom_', '').replace('_', ' ')
            co_symptoms.append({'name': human, 'count': int(val)})

    stats = {
        'dataset_source': _get_dataset_path(),
        'prevalence_pct': round(prevalence_pct, 2),
        'top_diseases': top_diseases,
        'co_symptoms': co_symptoms,
        'red_flags': RED_FLAGS_MAP.get(symptom, []),
        'advice': HOME_ADVICE_MAP.get(symptom, ['Monitor symptoms', 'Seek care if worsening'])
    }
    SYMPTOM_STATS_CACHE[symptom] = stats
    return stats

@app.route('/symptom_info', methods=['GET'])
def get_symptom_info():
    try:
        symptom = request.args.get('symptom', '').strip()
        if not symptom:
            return jsonify({'error': 'No symptom provided', 'symptom': symptom}), 400
        
        # Normalize symptom name to handle variations
        normalized_symptom = normalize_symptom_name(symptom)
        search_key = normalized_symptom or symptom.lower().strip()
        
        # Compute stats using the normalized name
        stats = compute_symptom_stats(search_key) or {}
        
        # Try to get cause - check both normalized and original
        cause = symptom_causes.get(search_key, None)
        if not cause:
            # Try variations
            variations = [symptom.lower(), symptom.lower().replace('_', ' '), symptom.lower().replace(' ', '_')]
            for var in variations:
                if var in symptom_causes:
                    cause = symptom_causes[var]
                    search_key = var
                    break
        
        # If still not found, provide a helpful default message
        if not cause:
            cause = f"This symptom may be related to various conditions. Please provide more details about the onset, duration, and severity. Consult a healthcare professional for accurate diagnosis."
        
        return jsonify({
            'symptom': symptom,
            'normalized_symptom': normalized_symptom or search_key,
            'cause': cause,
            'prevalence_pct': stats.get('prevalence_pct'),
            'top_diseases': stats.get('top_diseases', []),
            'co_symptoms': stats.get('co_symptoms', []),
            'red_flags': stats.get('red_flags', []),
            'advice': stats.get('advice', []),
            'dataset_source': stats.get('dataset_source')
        })
    except Exception as e:
        logger.error(f"Error in get_symptom_info: {str(e)}", exc_info=True)
        return jsonify({
            'error': str(e),
            'symptom': request.args.get('symptom', ''),
            'cause': 'Error retrieving symptom information. Please try again.',
            'prevalence_pct': None,
            'top_diseases': [],
            'co_symptoms': [],
            'red_flags': [],
            'advice': ['Consult a healthcare professional for accurate information.']
        }), 500

@app.route('/predict_by_symptoms', methods=['POST'])
def predict_by_symptoms():
    try:
        # Get data from request
        data = request.json
        symptoms = data.get('symptoms', [])
        
        # Make prediction
        prediction = symptom_model.predict(symptoms)[0]
        prediction_proba = symptom_model.predict_proba(symptoms)[0]
        
        # Get the highest probability
        max_proba_index = symptom_model.diseases.index(prediction)
        max_proba = prediction_proba[max_proba_index]
        
        # Create probabilities dictionary for all diseases
        probabilities = {}
        for i, disease in enumerate(symptom_model.diseases):
            probabilities[disease] = round(prediction_proba[i] * 100, 2)
        
        return jsonify({
            'disease': prediction,
            'confidence': round(max_proba * 100, 2),
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# __main__ block moved to end of file to ensure routes load
# ML symptom model artifacts (lazy-loaded)
ML_MODEL = None
ML_SYMPTOM_INDEX = []
ML_DISEASES = []


def load_ml_artifacts():
    global ML_MODEL, ML_SYMPTOM_INDEX, ML_DISEASES
    if ML_MODEL is not None:
        return
    try:
        import pickle, os
        with open(os.path.join(BASE_DIR, 'model','symptom_ml_artifacts.pkl'),'rb') as f:
            artifacts = pickle.load(f)
        ML_MODEL = artifacts.get('model')
        ML_SYMPTOM_INDEX = artifacts.get('symptom_index', [])
        ML_DISEASES = artifacts.get('diseases', [])
        print('Loaded ML symptom model artifacts:', len(ML_SYMPTOM_INDEX), 'symptoms', len(ML_DISEASES), 'diseases')
    except Exception as e:
        print('Failed to load ML symptom model artifacts:', e)


@app.route('/predict_by_symptoms_ml', methods=['POST'])
def predict_by_symptoms_ml():
    try:
        load_ml_artifacts()
        data = request.json
        symptoms = data.get('symptoms', [])

        # Fallback to rule-based model if ML model not available
        if ML_MODEL is None or not ML_SYMPTOM_INDEX:
            return predict_by_symptoms()

        # Build binary feature vector for selected symptoms
        x = np.zeros((1, len(ML_SYMPTOM_INDEX)))
        for s in symptoms:
            key = f"symptom_{s}"
            if key in ML_SYMPTOM_INDEX:
                idx = ML_SYMPTOM_INDEX.index(key)
                x[0, idx] = 1

        proba = ML_MODEL.predict_proba(x)[0]
        max_idx = int(np.argmax(proba))
        classes = ML_MODEL.classes_ if hasattr(ML_MODEL, 'classes_') else (ML_DISEASES or [])
        predicted = classes[max_idx] if (hasattr(classes, '__len__') and len(classes) > 0) else 'Unknown'

        probabilities = {classes[i]: round(float(proba[i] * 100), 2) for i in range(len(proba))} if (hasattr(classes, '__len__') and len(classes) > 0) else {}

        return jsonify({
            'disease': predicted,
            'confidence': round(float(proba[max_idx] * 100), 2),
            'probabilities': probabilities
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    try:
        import json, os
        path = os.path.join(BASE_DIR, 'model','symptoms_list.json')
        names = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                names = json.load(f)
        else:
            load_ml_artifacts()
            if ML_SYMPTOM_INDEX:
                names = [s.replace('symptom_', '') for s in ML_SYMPTOM_INDEX]
        return jsonify({'symptoms': names})
    except Exception as e:
        return jsonify({'error': str(e), 'symptoms': []}), 500

# Initialize logging from logging.conf if available
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logger = logging.getLogger('app')

# Enable CORS for API endpoints
# Enable CORS so the separately served frontend can call every endpoint.
# Restrict in production with e.g. CORS_ORIGINS="https://your-frontend.example".
_cors_origins = os.getenv('CORS_ORIGINS', '*')
CORS(app, resources={r"/*": {"origins": [o.strip() for o in _cors_origins.split(',')] if _cors_origins != '*' else '*'}})

@app.route('/health', methods=['GET'])
def health():
    """Lightweight health check endpoint."""
    # Try loading dataset and ml artifacts lazily
    df = _load_symptom_df()
    try:
        load_ml_artifacts()
    except Exception as e:
        logger.warning("ML artifacts load issue", extra={'error': str(e)})
    status = {
        'status': 'healthy',
        'uptime_seconds': round(time.time() - START_TIME, 2),
        'routes_count': len(app.url_map._rules),
        'dataset_loaded': bool(df is not None),
        'ml_model_loaded': bool(ML_MODEL is not None),
        'symptom_index_count': len(ML_SYMPTOM_INDEX) if ML_SYMPTOM_INDEX else 0,
        'diseases_count': len(ML_DISEASES) if ML_DISEASES else 0,
        'request_count_total': REQUEST_COUNTERS.get('total', 0)
    }
    return jsonify(status)

@app.route('/metrics', methods=['GET'])
def metrics():
    """Basic JSON metrics for observability without extra deps."""
    # Aggregate simple latency stats
    latency_summary = {}
    for path, values in ENDPOINT_LATENCIES.items():
        if values:
            latency_summary[path] = {
                'count': len(values),
                'min_ms': round(min(values) * 1000, 2),
                'max_ms': round(max(values) * 1000, 2),
                'avg_ms': round((sum(values) / len(values)) * 1000, 2)
            }
    data = {
        'uptime_seconds': round(time.time() - START_TIME, 2),
        'requests': dict(REQUEST_COUNTERS),
        'latencies': latency_summary
    }
    return jsonify(data)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        payload = request.get_json(silent=True) or {}
        message = str(payload.get('message', '')).lower().strip()
        if not message:
            return jsonify({
                'reply': 'Please describe your symptoms, duration, and severity.',
                'follow_up': ['List symptoms (e.g., fever, cough)', 'Duration in days', 'Severity (mild/moderate/severe)', 'Any red flags like chest pain or vision loss?']
            })

        # Load known symptom names
        names = []
        try:
            import json, os
            path = os.path.join(BASE_DIR, 'model','symptoms_list.json')
            if os.path.exists(path):
                with open(path, 'r') as f:
                    names = json.load(f)
            else:
                load_ml_artifacts()
                if ML_SYMPTOM_INDEX:
                    names = [s.replace('symptom_', '') for s in ML_SYMPTOM_INDEX]
        except Exception:
            names = list({k for k in symptom_causes.keys()})

        # Improved symptom extraction with normalization
        matched = []
        for name in names:
            key = name.lower()
            # Try exact match first
            if key and key in message:
                normalized = normalize_symptom_name(key)
                matched.append(normalized if normalized else key)
            # Also try with spaces instead of underscores
            elif key:
                key_with_spaces = key.replace('_', ' ')
                if key_with_spaces in message:
                    normalized = normalize_symptom_name(key)
                    matched.append(normalized if normalized else key)
        
        # Remove duplicates and limit
        matched = list(dict.fromkeys(matched))[:10]

        if not matched:
            return jsonify({
                'reply': 'I could not detect specific symptoms. Please try listing them clearly (e.g., "fever and cough for 2 days").',
                'suggestions': ['fever', 'cough', 'shortness of breath', 'chest pain', 'diarrhea', 'headache']
            })

        # Build aggregated insights
        insights = []
        for s in matched:
            normalized_s = normalize_symptom_name(s) or s
            stats = compute_symptom_stats(normalized_s) or {}
            cause = symptom_causes.get(normalized_s, None)
            if not cause:
                # Try to provide a generic response based on symptom name
                cause = f"This symptom may be related to various conditions. Please provide more details about the onset, duration, and severity."
            insights.append({
                'symptom': s,
                'normalized_symptom': normalized_s,
                'cause': cause,
                'prevalence_pct': stats.get('prevalence_pct'),
                'top_diseases': stats.get('top_diseases', []),
                'co_symptoms': stats.get('co_symptoms', []),
                'red_flags': stats.get('red_flags', []),
                'advice': stats.get('advice', [])
            })

        # Compose reply text
        parts = []
        for item in insights:
            part = f"For {item['symptom']}: {item['cause']}"
            if item.get('prevalence_pct') is not None:
                part += f" Prevalence: {item['prevalence_pct']}%."
            if item.get('top_diseases'):
                td = ', '.join([f"{d['name']} ({d['percent']}%)" for d in item['top_diseases'][:3]])
                part += f" Likely associated conditions: {td}."
            if item.get('red_flags'):
                rf = '; '.join(item['red_flags'][:3])
                part += f" Red flags: {rf}."
            if item.get('advice'):
                adv = '; '.join(item['advice'][:3])
                part += f" Home advice: {adv}."
            parts.append(part)

        follow_up = ['Duration and progression', 'Fever temperature (°C/°F)', 'Breathing difficulty severity', 'Chest pain description', 'Hydration and nutrition status']
        reply = ' '.join(parts)

        return jsonify({
            'reply': reply,
            'insights': insights,
            'next_steps': follow_up
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/symptom_catalog', methods=['GET'])
def symptom_catalog():
    try:
        # Build full catalog with stats
        import json, os
        names = []
        path = os.path.join(BASE_DIR, 'model','symptoms_list.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                names = json.load(f)
        else:
            load_ml_artifacts()
            if ML_SYMPTOM_INDEX:
                names = [s.replace('symptom_', '') for s in ML_SYMPTOM_INDEX]
            else:
                names = sorted(list({k for k in symptom_causes.keys()}))
        catalog = []
        for s in names:
            s_key = s.lower()
            stats = compute_symptom_stats(s_key) or {}
            catalog.append({
                'symptom': s_key,
                'cause': symptom_causes.get(s_key),
                'prevalence_pct': stats.get('prevalence_pct'),
                'top_diseases': stats.get('top_diseases', []),
                'co_symptoms': stats.get('co_symptoms', []),
                'red_flags': stats.get('red_flags', []),
                'advice': stats.get('advice', []),
                'dataset_source': stats.get('dataset_source')
            })
        return jsonify({'count': len(catalog), 'catalog': catalog})
    except Exception as e:
        return jsonify({'error': str(e), 'catalog': []}), 500

if __name__ == '__main__':
    # Run with env-based configuration
    port = int(os.getenv('PORT', '5000'))
    env = os.getenv('FLASK_ENV', 'development')
    debug = env != 'production'
    logger.info('Starting app', extra={'port': port, 'env': env})
    app.run(host='0.0.0.0', port=port, debug=debug)
