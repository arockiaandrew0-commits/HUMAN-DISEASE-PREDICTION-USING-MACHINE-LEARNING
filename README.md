# 🏥 Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning-powered web application that predicts diseases based on symptom analysis and basic health parameters. It combines a lightweight rule-based check on vital signs with a trained scikit-learn model for symptom-based prediction, plus an interactive chat-style assistant for symptom triage.

## 🌟 Features

### Core Functionality
- **Dual Prediction Modes**: Vitals-based heuristic check and ML-based symptom prediction
- **Real-time Predictions**: Instant disease prediction with confidence scores
- **Comprehensive Symptom Coverage**: 75 symptoms mapped to causes, and a 73-symptom ML feature set covering 21 disease categories
- **Modern Chat UI**: Single-page assistant with symptom chips, dark mode, and structured response cards (causes, prevalence, red flags, advice) instead of plain text
- **Balanced Training Data**: 42,000-row augmented dataset for the ML model

### Advanced Features
- **Symptom Information System**: Prevalence, co-occurring symptoms, red flags, and home-care advice per symptom, computed live from the dataset
- **Chat-style Triage**: Extracts symptoms from free text and returns aggregated guidance
- **Health & Metrics Endpoints**: Basic uptime and request-latency counters for monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlproject
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # edit .env with your own values — never commit this file
   ```

5. **(Optional) Retrain the model**

   Trained model artifacts already ship in `model/`, so this step isn't required to run the app.
   ```bash
   python train_symptom_model.py
   ```

6. **Run the app**
   ```bash
   python app.py
   ```

7. **Access the application**
   - Web app: `http://127.0.0.1:5000` (`/chat_ui` redirects here)

## 📊 Dataset Information

### Symptom Coverage
The `/symptoms` endpoint exposes 75 symptoms spanning respiratory, digestive, neurological, mental health, skin, and urinary categories (see `symptom_causes` in [app.py](app.py)). The ML training pipeline uses its own 73-symptom binary feature set (see `SYMPTOMS` in [train_symptom_model.py](train_symptom_model.py)).

### Disease Categories (21 conditions)
Common Cold, Influenza, COVID-19, Pneumonia, Bronchitis, Asthma, Allergy, Sinusitis, Migraine, Gastroenteritis, Food Poisoning, Irritable Bowel Syndrome, Hypertension, Diabetes, Urinary Tract Infection, Dermatitis, Anemia, Anxiety Disorder, Depressive Disorder, Arthritis, Healthy.

### Dataset Statistics
- **Total Samples**: 42,000 rows (`data/symptom_dataset_pro.csv`, 2,000 per disease class)
- **Features**: 73 binary symptom indicators
- **Validation Accuracy**: ~79% (Logistic Regression, 80/20 stratified split)
- **Strongest classes**: Influenza (F1 0.96), COVID-19 (F1 0.89), Common Cold (F1 0.88)
- **Weakest classes**: Healthy (F1 0.35, symptom overlap), Food Poisoning (F1 0.67)

## 🔧 API Reference

### Vitals-based Prediction
```http
POST /predict
Content-Type: application/json

{
    "age": 35,
    "temperature": 38.5,
    "heart_rate": 85,
    "blood_pressure": 120,
    "oxygen_level": 98,
    "pain_level": 3
}
```

**Response:**
```json
{
    "disease": "Flu",
    "confidence": 87.0
}
```

### Symptom-Based ML Prediction
```http
POST /predict_by_symptoms_ml
Content-Type: application/json

{
    "symptoms": ["fever", "cough", "headache", "fatigue"]
}
```

**Response:**
```json
{
    "disease": "Influenza",
    "confidence": 82.72,
    "probabilities": {
        "Influenza": 82.72,
        "COVID-19": 3.44,
        "Common Cold": 2.52
    }
}
```

### Symptom Information
```http
GET /symptom_info?symptom=fever
```

**Response:**
```json
{
    "symptom": "fever",
    "cause": "Fever is your body's way of fighting infection...",
    "prevalence_pct": 45.2,
    "top_diseases": [
        {"name": "Influenza", "percent": 89.5}
    ],
    "co_symptoms": ["headache", "fatigue", "chills"],
    "red_flags": ["Temperature above 40°C", "Persistent for >3 days"],
    "advice": ["Stay hydrated", "Rest", "Monitor temperature"]
}
```

### Other Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web interface |
| `/chat_ui` | GET | Redirects to `/` |
| `/chat` | POST | Chat-style symptom triage |
| `/symptoms` | GET | List of known symptoms |
| `/symptom_catalog` | GET | Full symptom catalog |
| `/predict_by_symptoms` | POST | Non-ML heuristic symptom prediction |
| `/health` | GET | Health check |
| `/metrics` | GET | Basic request/latency counters |

## 🧠 Machine Learning Pipeline

### Model Architecture
- **Algorithm**: Logistic Regression (`scikit-learn`)
- **Features**: 73 binary symptom indicators
- **Classes**: 21 disease categories
- **Validation**: Stratified 80/20 train/test split

### Training Process
1. **Data Loading**: Prefers `data/symptom_dataset_pro.csv`, falls back to `data/symptom_dataset.csv`, or synthesizes 10K rows if neither exists
2. **Training**: Logistic Regression fit on binary symptom features
3. **Evaluation**: Accuracy and per-class classification report printed to console
4. **Serialization**: Model, symptom index, and disease labels saved to `model/symptom_ml_artifacts.pkl`

Run `python train_symptom_model.py` to retrain — it overwrites `model/symptom_ml_artifacts.pkl`, `model/symptoms_list.json`, and `model/diseases_list.json` using the script's own symptom vocabulary, which differs from the one used elsewhere in the app.

## 📁 Project Structure

```
mlproject/
├── 📄 app.py                    # Main Flask application
├── 📄 train_symptom_model.py    # ML model training script
├── 📄 requirements.txt          # Python dependencies
├── 📄 logging.conf              # Logging configuration
├── 📄 README.md                 # Project documentation
├── 📁 data/
│   ├── symptom_dataset.csv          # Base synthetic dataset
│   ├── symptom_dataset_pro.csv      # Augmented dataset (used by default)
│   └── symptom_dataset_updated.csv
├── 📁 model/
│   ├── symptom_ml_artifacts.pkl # Trained model + preprocessing artifacts
│   ├── symptoms_list.json       # Symptom vocabulary
│   └── diseases_list.json       # Disease categories
├── 📁 static/                   # Assets served by Flask
├── 📁 templates/
│   └── index.html               # Single-page chat assistant UI
└── 📁 logs/                     # Rotating log files (created at runtime)
```

## 🔒 Security & Privacy

- No personal health information is persisted — predictions are stateless
- `.env` holds local configuration only and is excluded from version control
- Input validation on prediction endpoints; errors return JSON with a message, not stack traces

### Medical Disclaimer
⚠️ **IMPORTANT**: This application is for educational and demonstration purposes only. It should **not** be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 📦 Deployment

### Environment Variables
Copy `.env.example` to `.env`. Variables the app actually reads: `HOST`, `PORT`, `DEBUG`, `LOG_LEVEL`, `LOG_FILE_PATH`, `ERROR_LOG_FILE_PATH`. The example file also documents additional variables (database, Redis, cloud storage, etc.) for optional future integrations — none are required to run the app today.

### Production Setup (Gunicorn + Nginx)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
Put Nginx in front for HTTPS/reverse proxy, and run under a process manager (systemd, supervisord) for auto-restart.

### Observability
- Logs: `logs/` directory (`app.log`, `error.log`, `ml.log`, `api.log`, `security.log`, `flask.log`)
- Health: `GET /health` returns status and uptime
- Metrics: `GET /metrics` exposes basic request counters

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md).
