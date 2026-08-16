# 🏥 Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Flask web application that predicts likely diseases from either basic vital signs or a free-text/checklist description of symptoms. It combines a small rule-based heuristic for vitals with a trained scikit-learn model for symptom-based prediction, plus a lightweight in-browser chat UI that surfaces symptom information (typical causes, prevalence, red flags, home-care advice) drawn from the training dataset.

> ⚠️ **Medical disclaimer**: This project is for educational/demonstration purposes only. It is **not** a diagnostic tool and must not be used as a substitute for professional medical advice.

## Features

- **Symptom-based ML prediction** (`/predict_by_symptoms_ml`) — a scikit-learn model trained on a synthetic symptom dataset (21 disease classes, 73 symptoms) via [train_symptom_model.py](train_symptom_model.py)
- **Vitals-based prediction** (`/predict`) — a simple rule-based check using age, temperature, heart rate, blood pressure, oxygen level, and pain level
- **Symptom info lookup** (`/symptom_info`) — prevalence, co-occurring symptoms, red flags, and home-care advice per symptom, computed on the fly from the dataset
- **Chat-style assistant** (`/chat`, `/chat_ui`) — extracts symptoms from free text and returns aggregated guidance
- **Health & metrics endpoints** (`/health`, `/metrics`) — basic uptime/request counters for monitoring

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
git clone <your-repo-url>
cd mlproject
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
# edit .env with your own values — never commit this file
```

The trained model artifacts already ship in [model/](model/), so you can run the app directly. To retrain from scratch:

```bash
python train_symptom_model.py
```

Run the app:

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## API Reference

### Vitals-based prediction
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

### Symptom-based ML prediction
```http
POST /predict_by_symptoms_ml
Content-Type: application/json

{
    "symptoms": ["fever", "cough", "headache", "fatigue"]
}
```

### Symptom information
```http
GET /symptom_info?symptom=fever
```

### Other endpoints
- `GET /symptoms` — list of known symptoms
- `GET /symptom_catalog` — full symptom catalog
- `POST /predict_by_symptoms` — non-ML heuristic symptom prediction
- `POST /chat` — chat-style symptom triage
- `GET /chat_ui` — chat UI page
- `GET /health` — health check
- `GET /metrics` — basic request/latency counters

## Project Structure

```
mlproject/
├── app.py                     # Flask application (routes, prediction logic)
├── train_symptom_model.py     # Trains the symptom-based ML model
├── requirements.txt           # Python dependencies
├── logging.conf                # Logging configuration
├── data/
│   ├── symptom_dataset.csv           # Base symptom dataset
│   ├── symptom_dataset_pro.csv       # Augmented dataset (used by default if present)
│   └── symptom_dataset_updated.csv
├── model/
│   ├── symptom_ml_artifacts.pkl      # Trained model + preprocessing artifacts
│   ├── symptoms_list.json            # Symptom vocabulary
│   └── diseases_list.json            # Disease categories
├── static/                    # CSS/images served by Flask
├── templates/
│   ├── index.html             # Main web interface
│   └── chat.html              # Chat UI
└── logs/                      # Rotating log files (created at runtime)
```

## Machine Learning Pipeline

- **Data**: synthetic symptom profiles per disease (see `DISEASE_SYMPTOM_PROFILES` in [train_symptom_model.py](train_symptom_model.py)), sampled into a binary symptom-presence dataset
- **Model**: scikit-learn `LogisticRegression`, trained with an 80/20 train/test split
- **Artifacts**: trained model, symptom index, and label encoder are serialized to `model/symptom_ml_artifacts.pkl`

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Key variables actually used by the app:

| Variable | Purpose |
|---|---|
| `HOST`, `PORT` | Flask server bind address |
| `DEBUG` | Flask debug mode |
| `LOG_LEVEL`, `LOG_FILE_PATH`, `ERROR_LOG_FILE_PATH` | Logging configuration |

`.env.example` also documents additional variables (database, Redis, cloud storage, etc.) for future/optional integrations — the current codebase does not require them to run.

## License

MIT — see [LICENSE](LICENSE).
