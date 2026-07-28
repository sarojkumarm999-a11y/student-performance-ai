# 🎓 Student Performance Prediction & Analytics System

An end-to-end Machine Learning and Natural Language Processing (NLP) web application built to predict student academic performance, classify letter grades, identify at-risk students for early intervention, and generate personalized academic advising insights via OpenRouter & Anthropic LLMs.

---

## 🚀 Key Features

- **Multi-Target Predictive Machine Learning Pipeline**:
  - **GPA Regression**: Predicts final cumulative GPA on a **$0.00 - 10.00$ scale** using Random Forest Regressor ($R^2 > 0.99$, $MAE \approx 0.079$).
  - **Grade Tier Classification**: Multi-class classification ($A, B+, B, C, D, F$) via Gradient Boosting.
  - **3-Tier Academic Status Classification**:
    - 🟢 **`PASS`**: Predicted GPA $> 6.0$
    - 🟡 **`AT RISK`**: Predicted GPA between $4.5 - 6.0$
    - 🔴 **`FAIL`**: Predicted GPA $< 4.5$
- **Natural Language Processing (NLP)**:
  - Custom `SentimentFeatures` pipeline transformer utilizing **VADER SentimentIntensityAnalyzer** to derive numerical sentiment polarity (`compound`, `pos`, `neu`, `neg`) and text length metrics from qualitative behavioral notes.
- **Generative AI Academic Insights**:
  - OpenRouter API integration to output personalized 5-bullet action plans, risk flags, and supportive student guidance out-of-the-box (`Predict + AI Insights`).
- **1-Click Interactive Demo & REST API**:
  - Modern Flask web dashboard with dark-mode aesthetic, dynamic progress bars, glowing status badges, live model metrics visualizer, and 1-click **⚡ Fill Example** workflow.

---

## ⚡ Prediction Methodology & Fill Example Workflow

The application includes a **⚡ Fill Example** button to instantly test model inference with a realistic student profile:

### Sample Student Input Profile:
- **Age**: `19` years
- **Gender**: `Female`
- **Attendance**: `88.5%`
- **Study Hours / Day**: `4.5` hours
- **Previous GPA**: `8.30` / 10.00
- **Assignments Submitted**: `90%`
- **Extracurricular Score**: `7` / 10
- **Parent Education**: `Bachelor`
- **Internet Access**: `Yes (1)`
- **Part-Time Job**: `No (0)`
- **Counseling Sessions**: `2`
- **Behavioral & Observational Notes**: *"Highly engaged student, demonstrates active participation, attends tutoring regularly, and shows strong teamwork."*

### Prediction Output Results:
- 📈 **Predicted GPA**: `8.44 / 10.00`
- 🏷️ **Predicted Grade**: `B+`
- 🛡️ **Academic Status**: `PASS` (Green glowing status badge)
- 💡 **AI Insights Output**: Personalized 3-sentence performance summary, 5-bullet measurable action plan, 3 risk flags to monitor, and 2 encouragement lines.

---

## 📂 Project Architecture & Structure

```
student-performance-system/
├── app.py                      # Flask web server & REST API entry point
├── main.py                     # CLI quick inspection script
├── requirements.txt            # Python dependencies
├── Procfile                    # Production Cloud Deployment config (Gunicorn)
├── README.md                   # Project documentation
├── artifacts/                  # Trained models & metrics schema
│   ├── metrics.json            # Model evaluation metrics
│   ├── schema.json             # Feature schema metadata
│   ├── model_gpa.joblib        # Random Forest GPA model
│   ├── model_grade.joblib      # Gradient Boosting Grade model
│   └── model_pass_fail.joblib  # Gradient Boosting Status model
├── data/                       # Dataset directory
│   └── raw/
│       └── student_data.csv    # Rescaled 10-point student dataset
├── src/                        # Core ML & NLP pipeline modules
│   ├── __init__.py
│   ├── preprocess.py           # Data cleaning & VADER sentiment transformer
│   ├── predict.py              # Inference engine & 3-tier feature coercion
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
# Clone the repository
git clone https://github.com/sarojkumarm999-a11y/student-performance-ai.git
cd student-performance-ai

# Install required Python packages
pip install -r requirements.txt
```

### 3. API Key Configuration
The application comes with OpenRouter integration. You can set custom environment keys if desired:
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
👉 **`http://localhost:5000`**

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
      "behavioral_notes": "Highly engaged student, demonstrates active participation, attends tutoring regularly, and shows strong teamwork."
    }
  }
  ```
- **Response**:
  ```json
  {
    "predictions": {
      "predicted_gpa": 8.44,
      "grade": "B+",
      "pass_fail": 1
    }
  }
  ```

### 2. Prediction with AI Academic Insights
- **Endpoint**: `POST /api/predict-with-insights`
- **Response**: Includes `predictions` plus personalized `insights` narrative generated via OpenRouter LLM.

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
| **GPA Prediction** | Random Forest Regressor | Mean Absolute Error (MAE) | `~0.079` |
| **GPA Prediction** | Random Forest Regressor | $R^2$ Score | `> 0.99` |
| **Grade Tier Classification** | Gradient Boosting Classifier | Accuracy | `95.2%` |
| **Academic Status Tier (Pass/Risk/Fail)** | Gradient Boosting Classifier | Macro F1 Score | `0.941` |