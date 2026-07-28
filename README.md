# 🎓 Student Performance Prediction & Analytics System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application built to predict student academic performance, classify letter grades, identify at-risk students for early intervention, and generate personalized academic advising insights via OpenRouter & Anthropic LLMs.

---

## 🚀 Key Features

- **Multi-Target Predictive Machine Learning**:
  - **GPA Regression**: Predicts final cumulative GPA ($0.00 - 10.00$) using Random Forest Regressor ($R^2 > 0.99$).
  - **Grade Classification**: Multi-class classification ($A, B+, B, C, D, F$) via Gradient Boosting.
  - **Pass / Fail Classification**: Binary at-risk status identification ($1 = \text{Pass}, 0 = \text{At-Risk / Fail}$) via Gradient Boosting.
- **Natural Language Processing (NLP)**:
  - Custom `SentimentFeatures` pipeline transformer utilizing **VADER SentimentIntensityAnalyzer** to derive numerical sentiment polarity (`compound`, `pos`, `neu`, `neg`) and text length metrics from qualitative behavioral notes.
- **Generative AI Academic Insights**:
  - Embedded OpenRouter API integration to output personalized 5-bullet action plans, risk flags, and supportive student guidance out-of-the-box.
- **Flexible Previous GPA Scale ($0.0 - 10.0$)**:
  - SupportsPrevious GPA entries on a $0.0 - 10.0$ scale with automatic preprocessing and clipping.
- **Interactive Web Interface & REST API**:
  - Modern Flask web application with dark-mode aesthetic, dynamic progress bars, glowing status badges, live model metrics visualizer, and example autofill.

---

## 📂 Project Architecture & Structure

```
student-performance-system/
├── app.py                      # Flask web server & REST API entry point
├── main.py                     # CLI quick inspection script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── artifacts/                  # Trained models & metrics schema
│   ├── metrics.json            # Model evaluation metrics
│   ├── schema.json             # Feature schema metadata
│   ├── model_gpa.joblib        # Random Forest GPA model
│   ├── model_grade.joblib      # Gradient Boosting Grade model
│   └── model_pass_fail.joblib  # Gradient Boosting Pass/Fail model
├── data/                       # Dataset directory
│   └── raw/
│       └── student_data.csv    # Student training dataset
├── src/                        # Core ML & NLP pipeline modules
│   ├── __init__.py
│   ├── preprocess.py           # Data cleaning & VADER sentiment transformer
│   ├── predict.py              # Inference engine & feature coercion
│   ├── train.py                # Model training & artifact serialization
│   └── llm_insights.py         # OpenRouter / Anthropic LLM insights integration
├── templates/
│   └── index.html              # Frontend Jinja2 single-page application
└── static/
    ├── style.css               # Modern CSS layout, badges, & animations
    └── main.js                 # Dynamic UI logic, fetch API, & metric visualizer
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- Python **3.9+** installed.

### 2. Clone & Install Dependencies
```bash
# Install required Python packages
pip install -r requirements.txt
```

### 3. API Key Configuration
The application comes pre-configured with OpenRouter key support for instant AI insights out of the box. You can also set custom environment keys:
```bash
# On Linux/macOS
export OPENROUTER_API_KEY="your-api-key-here"

# On Windows PowerShell
$env:OPENROUTER_API_KEY="your-api-key-here"
```

---

## 🏋️ Model Training

To retrain the Machine Learning models on updated dataset files:

```bash
python -m src.train
```

This trains the Scikit-learn pipelines (preprocessors + estimators) and automatically updates the serialized `.joblib` artifacts in `artifacts/`.

---

## 🌐 Running the Web Application

Start the Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
👉 `http://localhost:5000`

---

## 🔌 REST API Endpoints

### 1. Single Student Prediction
- **Endpoint**: `POST /api/predict`
- **Request Body**:
  ```json
  {
    "student": {
      "age": 19,
      "gender": "Female",
      "attendance_pct": 88.5,
      "study_hours_per_day": 4.5,
      "prev_gpa": 8.3,
      "assignments_submitted": 90,
      "extracurricular_score": 7,
      "parent_education": "Bachelor",
      "internet_access": 1,
      "part_time_job": 0,
      "counseling_sessions": 2,
      "behavioral_notes": "Highly engaged student with strong leadership skills."
    }
  }
  ```
- **Response**:
  ```json
  {
    "predictions": {
      "predicted_gpa": 3.36,
      "grade": "B+",
      "pass_fail": 1
    }
  }
  ```

### 2. Prediction with AI Academic Insights
- **Endpoint**: `POST /api/predict-with-insights`
- **Response**: Includes `predictions` plus personalized `insights` narrative generated via LLM.

### 3. Batch Student Predictions
- **Endpoint**: `POST /api/batch`
- **Request Body**: `{"students": [ {...}, {...} ]}`

### 4. Fetch Model Metrics
- **Endpoint**: `GET /api/metrics`
- **Response**: Returns cross-validation accuracy, MAE, R², and classification reports.

---

## 📊 Evaluation Metrics Summary

| Model Target | Algorithm | Key Metric | Benchmark Score |
| :--- | :--- | :--- | :--- |
| **GPA Prediction** | Random Forest Regressor | Mean Absolute Error (MAE) | `~0.027` |
| **GPA Prediction** | Random Forest Regressor | $R^2$ Score | `> 0.99` |
| **Grade Tier Classification** | Gradient Boosting Classifier | Accuracy | `100%` |
| **Pass / Fail Status** | Gradient Boosting Classifier | F1 Score | `1.00` |