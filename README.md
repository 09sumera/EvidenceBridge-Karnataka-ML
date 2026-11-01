# EvidenceBridge Karnataka ML

**AI/ML-powered hospital evidence analysis and treatment package cost prediction platform for Karnataka, India.**

## Project Overview

EvidenceBridge combines machine learning, rule-based scoring, geospatial validation and hospital pricing data.

## Key Features

### 1. Hospital Anomaly Detection
- Isolation Forest model.
- Identifies unusual hospital records for manual review.

### 2. Evidence Completeness Scoring
- Rule-based hospital evidence completeness assessment.
- HIGH, MODERATE, LOW and VERY_LOW categories.

### 3. Hospital Recommendation Support
- Rule-based filtering by district and specialty.

### 4. Treatment Package Cost Prediction
- Random Forest Regressor.
- Uses procedure/package name and room category.
- Experimental estimates based on Sakhii Hospital, Bengaluru pricing data.
- Not a validated Karnataka-wide cost prediction model.

### 5. Hospital Location Validation
- Geographical boundary and coordinate validation.
- Identifies records requiring manual review.

## Technology Stack
- Python
- Flask
- Scikit-learn
- Pandas and NumPy
- Google Colab
- Hugging Face Hub
- Git and GitHub
- Gunicorn

## Project Structure
```text
EvidenceBridge-Karnataka-ML/
├── app.py
├── requirements.txt
├── config/
├── src/
├── data/       (excluded from GitHub)
├── models/     (stored separately)
├── notebooks/
├── reports/
└── tests/
```

## Hugging Face Model
https://huggingface.co/sumera09/evidencebridge-karnataka-ml

## API Endpoints
| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | API information |
| GET | /api/health | Health check |
| GET | /api/cost/health | Cost model health |
| POST | /api/cost/predict | Treatment cost estimation |

## Installation
```bash
git clone https://github.com/09sumera/EvidenceBridge-Karnataka-ML.git
cd EvidenceBridge-Karnataka-ML
pip install -r requirements.txt
python app.py
```

## Deployment Architecture
- Google Colab: Model development
- Hugging Face: Model storage
- GitHub: Source code
- Render: Flask API hosting

## Limitations and Privacy
- Raw hospital and patient-cost datasets are excluded.
- Anomaly flags require manual review.
- Evidence completeness does not establish clinical quality.
- Cost estimates are experimental and limited to supported packages.

## Author
**Sumera Anjum**

GitHub: https://github.com/09sumera
