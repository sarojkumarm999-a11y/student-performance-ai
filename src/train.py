from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import REQUIRED_FEATURES, SentimentFeatures, clean_student_data, load_student_data


@dataclass
class TrainConfig:
    csv_path: str = "data/raw/student_data.csv"
    artifacts_dir: str = "artifacts"
    test_size: float = 0.2
    random_state: int = 42

    model_gpa_path: str = "artifacts/model_gpa.joblib"
    model_grade_path: str = "artifacts/model_grade.joblib"
    model_pass_fail_path: str = "artifacts/model_pass_fail.joblib"
    metrics_path: str = "artifacts/metrics.json"
    schema_path: str = "artifacts/schema.json"


def _build_preprocessor() -> ColumnTransformer:
    numeric_features = [
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
    categorical_features = ["gender", "parent_education"]
    text_feature = ["behavioral_notes"]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("sent", SentimentFeatures(text_col="behavioral_notes")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
            ("txt", text_pipe, text_feature),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _build_models(random_state: int) -> Tuple[Pipeline, Pipeline, Pipeline]:
    preprocessor = _build_preprocessor()

    gpa_model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
    )
    grade_model = GradientBoostingClassifier(random_state=random_state)
    pass_fail_model = GradientBoostingClassifier(random_state=random_state)

    gpa_pipe = Pipeline(steps=[("prep", preprocessor), ("model", gpa_model)])
    grade_pipe = Pipeline(steps=[("prep", preprocessor), ("model", grade_model)])
    pass_fail_pipe = Pipeline(steps=[("prep", preprocessor), ("model", pass_fail_model)])

    return gpa_pipe, grade_pipe, pass_fail_pipe


def _ensure_artifacts_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def train_models(config: TrainConfig | None = None) -> Dict[str, Any]:
    cfg = config or TrainConfig()
    _ensure_artifacts_dir(cfg.artifacts_dir)

    df = clean_student_data(load_student_data(cfg.csv_path))

    X = df[REQUIRED_FEATURES].copy()
    y_gpa = pd.to_numeric(df["predicted_gpa"], errors="coerce")
    y_grade = df["grade"].astype(str)
    y_pass = pd.to_numeric(df["pass_fail"], errors="coerce").fillna(0).astype(int)

    X_train, X_test, yg_train, yg_test, ygr_train, ygr_test, yp_train, yp_test = train_test_split(
        X,
        y_gpa,
        y_grade,
        y_pass,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y_pass if len(np.unique(y_pass)) > 1 else None,
    )

    model_gpa, model_grade, model_pass = _build_models(cfg.random_state)

    model_gpa.fit(X_train, yg_train)
    model_grade.fit(X_train, ygr_train)
    model_pass.fit(X_train, yp_train)

    pred_gpa = model_gpa.predict(X_test)
    pred_grade = model_grade.predict(X_test)
    pred_pass = model_pass.predict(X_test)

    metrics: Dict[str, Any] = {
        "data": {
            "rows": int(len(df)),
            "test_size": float(cfg.test_size),
            "random_state": int(cfg.random_state),
        },
        "gpa": {
            "mae": float(mean_absolute_error(yg_test, pred_gpa)),
            "rmse": float(root_mean_squared_error(yg_test, pred_gpa)),
            "r2": float(r2_score(yg_test, pred_gpa)),
        },
        "grade": {
            "accuracy": float(accuracy_score(ygr_test, pred_grade)),
            "report": classification_report(ygr_test, pred_grade, output_dict=True, zero_division=0),
            "labels": sorted(list({str(x) for x in y_grade.unique()})),
        },
        "pass_fail": {
            "accuracy": float(accuracy_score(yp_test, pred_pass)),
            "f1": float(f1_score(yp_test, pred_pass, average="macro", zero_division=0)),
        },
    }

    joblib.dump(model_gpa, cfg.model_gpa_path)
    joblib.dump(model_grade, cfg.model_grade_path)
    joblib.dump(model_pass, cfg.model_pass_fail_path)

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    schema = {
        "required_features": REQUIRED_FEATURES,
        "targets": ["predicted_gpa", "grade", "pass_fail"],
        "artifacts": {
            "model_gpa": cfg.model_gpa_path,
            "model_grade": cfg.model_grade_path,
            "model_pass_fail": cfg.model_pass_fail_path,
            "metrics": cfg.metrics_path,
        },
    }
    with open(cfg.schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_models()
    print(json.dumps(m, indent=2))
