from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .preprocess import REQUIRED_FEATURES


BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Artifacts:
    model_gpa_path: str = str(BASE_DIR / "artifacts" / "model_gpa.joblib")
    model_grade_path: str = str(BASE_DIR / "artifacts" / "model_grade.joblib")
    model_pass_fail_path: str = str(BASE_DIR / "artifacts" / "model_pass_fail.joblib")
    metrics_path: str = str(BASE_DIR / "artifacts" / "metrics.json")


class Predictor:
    def __init__(self, artifacts: Artifacts | None = None):
        self.artifacts = artifacts or Artifacts()
        self._model_gpa = None
        self._model_grade = None
        self._model_pass = None

    def load(self) -> "Predictor":
        try:
            self._model_gpa = joblib.load(self.artifacts.model_gpa_path)
            self._model_grade = joblib.load(self.artifacts.model_grade_path)
            self._model_pass = joblib.load(self.artifacts.model_pass_fail_path)
        except Exception as e:
            # Self-healing: if joblib unpickling fails due to environment/version differences, retrain models dynamically
            from .train import train_models

            train_models()
            self._model_gpa = joblib.load(self.artifacts.model_gpa_path)
            self._model_grade = joblib.load(self.artifacts.model_grade_path)
            self._model_pass = joblib.load(self.artifacts.model_pass_fail_path)
        return self

    def _ensure_loaded(self) -> None:
        if self._model_gpa is None or self._model_grade is None or self._model_pass is None:
            self.load()

    @staticmethod
    def _to_frame(student: Dict[str, Any] | List[Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(student, list):
            df = pd.DataFrame(student)
        else:
            df = pd.DataFrame([student])

        for col in REQUIRED_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        df = df[REQUIRED_FEATURES].copy()

        numeric_cols = [
            "age",
            "attendance_pct",
            "study_hours_per_day",
            "prev_gpa",
            "assignments_submitted",
            "extracurricular_score",
            "counseling_sessions",
            "internet_access",
            "part_time_job",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["behavioral_notes"] = df["behavioral_notes"].fillna("").astype(str)
        return df

    def predict_one(self, student: Dict[str, Any]) -> Dict[str, Any]:
        out = self.predict_batch([student])
        return out[0]

    def predict_batch(self, students: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        X = self._to_frame(students)

        gpa = self._model_gpa.predict(X)
        grade = self._model_grade.predict(X)
        pass_fail = self._model_pass.predict(X)

        results: List[Dict[str, Any]] = []
        for i in range(len(X)):
            predicted_gpa = float(np.clip(gpa[i], 0.0, 10.0))
            if predicted_gpa > 6.0:
                status = 1  # PASS
            elif predicted_gpa >= 4.5:
                status = 2  # AT RISK
            else:
                status = 0  # FAIL

            results.append(
                {
                    "predicted_gpa": predicted_gpa,
                    "grade": str(grade[i]),
                    "pass_fail": status,
                }
            )
        return results

    def load_metrics(self) -> Dict[str, Any]:
        p = Path(self.artifacts.metrics_path)
        if not p.exists():
            return {"error": "metrics_not_found", "path": str(p)}
        return json.loads(p.read_text(encoding="utf-8"))

