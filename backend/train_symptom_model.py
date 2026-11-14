import os
import random
import json
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Expanded symptoms list (50+ common ones)
SYMPTOMS: List[str] = [
    'headache','fever','cough','sore_throat','fatigue','nausea','dizziness','chest_pain',
    'shortness_of_breath','abdominal_pain','muscle_aches','joint_pain','rash','vomiting','diarrhea','chills',
    'loss_of_smell','loss_of_taste','runny_nose','congestion','sneezing','ear_pain','sinus_pressure','back_pain',
    'anxiety','depression','insomnia','palpitations','night_sweats','weight_loss','weight_gain','itching',
    'wheezing','blurred_vision','photophobia','body_ache','burning_urination','frequent_urination','blood_in_stool',
    'constipation','bloating','heartburn','indigestion','swollen_lymph_nodes','mouth_ulcers','dry_cough',
    'productive_cough','yellow_phlegm','green_phlegm'
]

# Diseases list (20 common ones)
DISEASES: List[str] = [
    'Common Cold','Influenza','COVID-19','Pneumonia','Bronchitis','Asthma','Allergy','Sinusitis','Migraine',
    'Gastroenteritis','Food Poisoning','Irritable Bowel Syndrome','Hypertension','Diabetes','Urinary Tract Infection',
    'Dermatitis','Anemia','Anxiety Disorder','Depressive Disorder','Arthritis','Healthy'
]

# Symptom profiles: probabilities of symptoms being present for each disease
DISEASE_SYMPTOM_PROFILES: Dict[str, Dict[str, float]] = {
    'Common Cold': {
        'runny_nose': 0.8,'sore_throat': 0.6,'sneezing': 0.7,'congestion': 0.7,'cough': 0.6,'fever': 0.3,'fatigue': 0.5
    },
    'Influenza': {
        'fever': 0.85,'chills': 0.8,'fatigue': 0.8,'muscle_aches': 0.75,'body_ache': 0.8,'cough': 0.6,'headache': 0.5
    },
    'COVID-19': {
        'fever': 0.7,'dry_cough': 0.75,'loss_of_smell': 0.6,'loss_of_taste': 0.6,'shortness_of_breath': 0.4,'fatigue': 0.7
    },
    'Pneumonia': {
        'fever': 0.8,'cough': 0.7,'shortness_of_breath': 0.7,'chest_pain': 0.6,'yellow_phlegm': 0.6,'green_phlegm': 0.4
    },
    'Bronchitis': {
        'productive_cough': 0.8,'green_phlegm': 0.6,'wheezing': 0.5,'cough': 0.7,'fatigue': 0.5
    },
    'Asthma': {
        'wheezing': 0.8,'shortness_of_breath': 0.7,'cough': 0.6,'chest_pain': 0.4
    },
    'Allergy': {
        'sneezing': 0.8,'itching': 0.7,'runny_nose': 0.7,'rash': 0.5,'congestion': 0.6
    },
    'Sinusitis': {
        'sinus_pressure': 0.8,'congestion': 0.7,'headache': 0.6,'ear_pain': 0.5,'sore_throat': 0.4
    },
    'Migraine': {
        'headache': 0.9,'photophobia': 0.7,'nausea': 0.6,'vomiting': 0.4,'dizziness': 0.5
    },
    'Gastroenteritis': {
        'vomiting': 0.8,'diarrhea': 0.85,'abdominal_pain': 0.7,'fever': 0.4,'chills': 0.5
    },
    'Food Poisoning': {
        'vomiting': 0.85,'diarrhea': 0.8,'abdominal_pain': 0.75,'fever': 0.4
    },
    'Irritable Bowel Syndrome': {
        'abdominal_pain': 0.8,'bloating': 0.8,'constipation': 0.6,'diarrhea': 0.5
    },
    'Hypertension': {
        'headache': 0.5,'dizziness': 0.5,'blurred_vision': 0.4,'palpitations': 0.4
    },
    'Diabetes': {
        'frequent_urination': 0.7,'weight_loss': 0.5,'fatigue': 0.5
    },
    'Urinary Tract Infection': {
        'burning_urination': 0.85,'frequent_urination': 0.75,'fever': 0.4,'abdominal_pain': 0.4
    },
    'Dermatitis': {
        'rash': 0.8,'itching': 0.8
    },
    'Anemia': {
        'fatigue': 0.75,'dizziness': 0.6,'shortness_of_breath': 0.4
    },
    'Anxiety Disorder': {
        'anxiety': 0.9,'palpitations': 0.6,'insomnia': 0.7,'night_sweats': 0.4
    },
    'Depressive Disorder': {
        'depression': 0.9,'fatigue': 0.6,'insomnia': 0.6,'weight_gain': 0.4
    },
    'Arthritis': {
        'joint_pain': 0.85,'muscle_aches': 0.6
    },
    'Healthy': {
        # Low probabilities across the board
        'fatigue': 0.1,'headache': 0.1
    }
}

# Prior distribution for diseases (approximate prevalence)
DISEASE_PRIORS: Dict[str, float] = {
    'Common Cold': 0.08,'Influenza': 0.06,'COVID-19': 0.05,'Pneumonia': 0.03,'Bronchitis': 0.04,
    'Asthma': 0.04,'Allergy': 0.07,'Sinusitis': 0.05,'Migraine': 0.05,'Gastroenteritis': 0.05,
    'Food Poisoning': 0.05,'Irritable Bowel Syndrome': 0.04,'Hypertension': 0.05,'Diabetes': 0.04,
    'Urinary Tract Infection': 0.04,'Dermatitis': 0.03,'Anemia': 0.03,'Anxiety Disorder': 0.03,
    'Depressive Disorder': 0.03,'Arthritis': 0.03,'Healthy': 0.1
}

# Normalize priors
_total = sum(DISEASE_PRIORS.values())
DISEASE_PRIORS = {d: p/_total for d, p in DISEASE_PRIORS.items()}


def create_synthetic_symptom_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs('data', exist_ok=True)

    diseases = list(DISEASE_PRIORS.keys())
    prior_probs = np.array([DISEASE_PRIORS[d] for d in diseases])

    rows = []
    for _ in range(n_samples):
        # Sample a disease according to prior
        disease = np.random.choice(diseases, p=prior_probs)
        profile = DISEASE_SYMPTOM_PROFILES.get(disease, {})

        symptom_flags = {}
        # Generate symptoms based on disease profile + small noise
        for s in SYMPTOMS:
            base = profile.get(s, 0.05)
            noise = np.clip(np.random.normal(0, 0.02), -0.02, 0.02)
            p = np.clip(base + noise, 0.0, 0.95)
            symptom_flags[s] = 1 if np.random.rand() < p else 0
        # Ensure at least one symptom
        if sum(symptom_flags.values()) == 0:
            # Randomly pick 1-2 symptoms
            idxs = np.random.choice(len(SYMPTOMS), size=np.random.randint(1, 3), replace=False)
            for i in idxs:
                symptom_flags[SYMPTOMS[i]] = 1

        row = {f'symptom_{s}': symptom_flags[s] for s in SYMPTOMS}
        row['disease'] = disease
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('data/symptom_dataset.csv', index=False)
    print(f"Saved synthetic dataset to data/symptom_dataset.csv with {len(df)} rows, {len(SYMPTOMS)} symptoms.")
    return df


def train_symptom_model(df: pd.DataFrame):
    os.makedirs('model', exist_ok=True)

    X = df.drop('disease', axis=1).values
    y = df['disease'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000, n_jobs=None)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.3f}")
    print("Classification report (head):")
    print(classification_report(y_test, y_pred, zero_division=0))

    artifacts = {
        'model': model,
        'symptom_index': [col for col in df.columns if col.startswith('symptom_')],
        'diseases': sorted(set(y))
    }
    with open('model/symptom_ml_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    print("Saved model artifacts to model/symptom_ml_artifacts.pkl")

    # Also save JSON lists for reference / front-end use
    with open('model/symptoms_list.json', 'w') as f:
        json.dump(SYMPTOMS, f, indent=2)
    with open('model/diseases_list.json', 'w') as f:
        json.dump(DISEASES, f, indent=2)

    return artifacts


if __name__ == '__main__':
    # Read/write data/ and model/ next to this script regardless of where it is run from
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # Prefer the augmented pro dataset if available, otherwise generate synthetic
    pro_path = os.path.join('data', 'symptom_dataset_pro.csv')
    base_path = os.path.join('data', 'symptom_dataset.csv')

    df = None
    if os.path.exists(pro_path):
        print(f"Loading augmented pro dataset: {pro_path}")
        df = pd.read_csv(pro_path)
    elif os.path.exists(base_path):
        print(f"Loading existing dataset: {base_path}")
        df = pd.read_csv(base_path)
    else:
        print('No dataset found, creating 10k synthetic symptom–disease dataset...')
        df = create_synthetic_symptom_dataset(n_samples=10000, seed=42)

    print('Training ML model on selected dataset...')
    train_symptom_model(df)
    print('Done.')