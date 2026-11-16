# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0] - 2026-11-14

### Added
- Redesigned interface: Symptom Checker, Assistant, Vitals Check and About pages with light/dark themes and mobile navigation
- Separate `frontend/` (static HTML/CSS/JS) and `backend/` (Flask API); the backend also serves the frontend
- Vitals Check page using the `/predict` endpoint
- Prediction responses report which selected symptoms the model does not use
- Security headers, request size limit and input validation with `400` errors
- Request metrics collection for `/metrics`; rotating log file in `backend/logs/`
- `run.bat` one-click launcher, favicon and social/SEO metadata

### Changed
- Debug mode is off by default (`DEBUG=true` to enable); `HOST`, `PORT`, `LOG_LEVEL`, `CORS_ORIGINS` are read from the environment
- Error responses no longer expose exception details or server file paths
- Dependencies updated to versions that install on current Python (3.11+)

### Removed
- Unused `logging.conf`, unused settings from `.env.example`, legacy `/chat_ui` route and old templates

## [1.0.0] - 2026-08-16

### Added
- Flask web app with vitals-based and symptom-based disease prediction
- Symptom-based ML model (scikit-learn) trained on a synthetic symptom dataset
- Symptom information lookup with prevalence, co-symptoms, red flags, and home-care advice
- Chat-style symptom triage endpoint and UI
- Health and metrics endpoints for basic monitoring
