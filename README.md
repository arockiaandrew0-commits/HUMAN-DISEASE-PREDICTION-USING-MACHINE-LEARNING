# 🏥 Disease Prediction System

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MedPredict is a web application that predicts likely diseases from symptoms and basic vital signs. It combines a trained scikit-learn model for symptom-based prediction with a rule-based vital-signs screen and a chat-style assistant that explains causes, warning signs and self-care advice for each symptom.

## 🌟 Features

### Core Functionality
- **Dual Prediction Modes**: Vitals-based heuristic check and ML-based symptom prediction
- **Real-time Predictions**: Instant disease prediction with confidence scores
- **Symptom Coverage**: 73 selectable symptoms with guidance; the ML model uses 49 of them as features across 21 disease categories (the result card says which selected symptoms the model did not use)
- **Responsive Interface**: Symptom Checker, Assistant, Vitals Check and About pages with light/dark themes, mobile navigation and accessible controls
- **Balanced Training Data**: 42,000-row augmented dataset for the ML model

### Advanced Features
- **Symptom Information System**: Prevalence, co-occurring symptoms, red flags, and home-care advice per symptom, computed live from the dataset
- **Chat-style Triage**: Extracts symptoms from free text and returns aggregated guidance
- **Health & Metrics Endpoints**: Basic uptime and request-latency counters for monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
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
   pip install -r backend/requirements.txt
   ```

4. **(Optional) Configure**

   Defaults work out of the box. The settings the backend reads (`HOST`, `PORT`, `DEBUG`, `LOG_LEVEL`, `CORS_ORIGINS`) are documented in [.env.example](.env.example); set them as environment variables before starting the backend.

5. **(Optional) Retrain the model**

   Trained model artifacts already ship in `backend/model/`, so this step isn't required to run the app.
   ```bash
   python backend/train_symptom_model.py
   ```

6. **Run the app**

   **Windows, one click:** double-click `run.bat`. It uses the project's `.venv` (creating it and installing packages if needed), starts the backend and opens `http://localhost:5000`. Keep its window open while you use the app.

   **Or from a terminal** — use the virtual-environment Python, not the system `python`, which may not have Flask installed:
   ```bash
   .venv\Scripts\python.exe backend\app.py
   ```
   (macOS/Linux: `.venv/bin/python backend/app.py`)

7. **Open the application** at `http://localhost:5000`.

   The backend serves both the website (from `frontend/`) and the JSON API, so one command is enough. `GET /api` lists the endpoints.

   **Hosting the frontend separately** (optional): serve `frontend/` from any static server, e.g. `python -m http.server 8000 --directory frontend`, and keep the backend running on port 5000. Set `API_BASE_URL` in `frontend/js/config.js` if the backend lives elsewhere, and `CORS_ORIGINS` on the backend to match.

   **If no symptoms appear or answers never arrive**, the backend isn't running. Start it as above and reload the page.

## 📊 Dataset Information

### Symptom Coverage
The `/symptoms` endpoint exposes 73 symptoms (from `backend/model/symptoms_list.json`) spanning respiratory, digestive, neurological, mental health, skin, and urinary categories; causes and advice live in [backend/app.py](backend/app.py). The bundled model was trained on the 49 symptom columns of the bundled dataset, so prediction responses list any selected symptoms the model does not use in `symptoms_not_in_model`.

### Disease Categories (21 conditions)
Common Cold, Influenza, COVID-19, Pneumonia, Bronchitis, Asthma, Allergy, Sinusitis, Migraine, Gastroenteritis, Food Poisoning, Irritable Bowel Syndrome, Hypertension, Diabetes, Urinary Tract Infection, Dermatitis, Anemia, Anxiety Disorder, Depressive Disorder, Arthritis, Healthy.

### Dataset Statistics
- **Total Samples**: 42,000 rows (`backend/data/symptom_dataset_pro.csv`, 2,000 per disease class)
- **Features**: 49 binary symptom indicators
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
    },
    "symptoms_used": ["fever", "cough", "headache", "fatigue"],
    "symptoms_not_in_model": []
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
| `/` | GET | Web interface (served from `frontend/`) |
| `/api` | GET | List of API endpoints |
| `/chat` | POST | Chat-style symptom triage |
| `/symptoms` | GET | List of known symptoms |
| `/symptom_catalog` | GET | Full symptom catalog |
| `/predict_by_symptoms` | POST | Non-ML heuristic symptom prediction |
| `/health` | GET | Health check |
| `/metrics` | GET | Request counts and latency per endpoint |

Invalid input returns HTTP `400` with `{"error": "<message>"}` (for example a missing vital sign or an out-of-range value); unexpected failures return `500` with a generic message and are logged on the server.

## 🧠 Machine Learning Pipeline

### Model Architecture
- **Algorithm**: Logistic Regression (`scikit-learn`)
- **Features**: 49 binary symptom indicators (bundled model)
- **Classes**: 21 disease categories
- **Validation**: Stratified 80/20 train/test split

### Training Process
1. **Data Loading**: Prefers `backend/data/symptom_dataset_pro.csv`, falls back to `backend/data/symptom_dataset.csv`, or synthesizes 10K rows if neither exists
2. **Training**: Logistic Regression fit on binary symptom features
3. **Evaluation**: Accuracy and per-class classification report printed to console
4. **Serialization**: Model, symptom index, and disease labels saved to `backend/model/symptom_ml_artifacts.pkl`

Run `python backend/train_symptom_model.py` to retrain — it overwrites `model/symptom_ml_artifacts.pkl`, `model/symptoms_list.json`, and `model/diseases_list.json` using the script's own symptom vocabulary, which differs from the one used elsewhere in the app.

## 📁 Project Structure

```
mlproject/
├── 📁 backend/                  # Flask REST API (Python)
│   ├── app.py                   # API routes, prediction logic, serves frontend/
│   ├── train_symptom_model.py   # ML model training script
│   ├── requirements.txt         # Python dependencies
│   ├── 📁 data/                 # Symptom–disease datasets (CSV)
│   ├── 📁 model/                # Trained model + symptom/disease lists
│   └── 📁 logs/                 # Log files (created at runtime)
├── 📁 frontend/                 # Static web UI (no build step)
│   ├── index.html               # Page markup: checker, assistant, vitals, about
│   ├── favicon.svg
│   ├── 📁 css/styles.css        # Design tokens, layout, light/dark themes
│   └── 📁 js/
│       ├── theme.js             # Applies saved/system theme before first paint
│       ├── config.js            # Backend API URL
│       └── app.js               # Routing, API calls, rendering
├── 📄 run.bat                   # Sets up .venv and starts the app on Windows
├── 📄 .env.example              # Documented configuration settings
└── 📄 README.md
```

## 🔒 Security & Privacy

- No personal health information is persisted — predictions are stateless
- Debug mode (interactive debugger) is off by default; enable it only on a private machine with `DEBUG=true`
- Input validation on all POST endpoints, a 64 KB request size limit, and JSON errors without stack traces or server paths
- Security headers on every response (`Content-Security-Policy` on pages, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`)
- Restrict cross-origin access with `CORS_ORIGINS` when the frontend is hosted separately

### Medical Disclaimer
⚠️ **IMPORTANT**: This application is for educational and demonstration purposes only. It should **not** be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 📦 Deployment

### Environment Variables
| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Listen address (`127.0.0.1` keeps it local to the machine) |
| `PORT` | `5000` | Listen port |
| `DEBUG` | `false` | Auto-reload and interactive debugger — development only |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CORS_ORIGINS` | `*` | Comma-separated origins allowed to call the API |

Windows example: `set PORT=8080` then `.venv\Scripts\python.exe backend\app.py`.

### Production Setup
`python backend/app.py` uses Flask's built-in server, which is fine for demos. For a public deployment run the app under a WSGI server (installed separately), e.g. on Linux:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 --chdir backend app:app
```
The backend serves the frontend itself; alternatively serve `frontend/` as static files from Nginx (or any static host) and proxy the API, setting `API_BASE_URL` in `frontend/js/config.js` and `CORS_ORIGINS` on the backend to match. Use Nginx for HTTPS, and run under a process manager (systemd, supervisord) for auto-restart.

### Observability
- Logs: console and `backend/logs/app.log` (rotated at 5 MB, 3 backups)
- Health: `GET /health` returns status and uptime
- Metrics: `GET /metrics` exposes request counts and min/avg/max latency per endpoint (in memory, resets on restart)

## 📝 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md).
