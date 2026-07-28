from __future__ import annotations

import os
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

from src.llm_insights import generate_student_insights
from src.predict import Predictor
from src.preprocess import REQUIRED_FEATURES


def _extract_student_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "student" in payload and isinstance(payload["student"], dict):
        student = payload["student"]
    else:
        student = payload

    return {k: student.get(k) for k in REQUIRED_FEATURES}


def create_app() -> Flask:
    app = Flask(__name__)
    predictor = Predictor()

    @app.get("/")
    def index():
        return render_template("index.html", required_features=REQUIRED_FEATURES)

    @app.get("/api/metrics")
    def api_metrics():
        return jsonify(predictor.load_metrics())

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(force=True, silent=False) or {}
        student = _extract_student_payload(payload)
        pred = predictor.predict_one(student)
        return jsonify({"student": student, "predictions": pred})

    @app.post("/api/batch")
    def api_batch():
        payload = request.get_json(force=True, silent=False) or {}
        students = payload.get("students", [])
        if not isinstance(students, list):
            return jsonify({"error": "students_must_be_a_list"}), 400

        cleaned = [_extract_student_payload({"student": s}) for s in students]
        preds = predictor.predict_batch(cleaned)
        return jsonify(
            {
                "count": len(cleaned),
                "results": [{"student": s, "predictions": p} for s, p in zip(cleaned, preds)],
            }
        )

    @app.post("/api/predict-with-insights")
    def api_predict_with_insights():
        payload = request.get_json(force=True, silent=False) or {}
        student = _extract_student_payload(payload)
        pred = predictor.predict_one(student)

        insights: str | None = None
        insights_error: str | None = None

        try:
            insights = generate_student_insights(student=student, predictions=pred)
        except Exception as e:
            insights_error = str(e)

        return jsonify(
            {
                "student": student,
                "predictions": pred,
                "insights": insights,
                "insights_error": insights_error,
            }
        )

    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    create_app().run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
