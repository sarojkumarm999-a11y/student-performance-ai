from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REQUIRED_FEATURES: List[str] = [
    "age",
    "gender",
    "attendance_pct",
    "study_hours_per_day",
    "prev_gpa",
    "assignments_submitted",
    "extracurricular_score",
    "parent_education",
    "internet_access",
    "part_time_job",
    "counseling_sessions",
    "behavioral_notes",
]

TARGETS: List[str] = ["predicted_gpa", "grade", "pass_fail"]


def load_student_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in REQUIRED_FEATURES + TARGETS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=TARGETS).reset_index(drop=True)

    df["behavioral_notes"] = df["behavioral_notes"].fillna("").astype(str)

    def _clip(col: str, low: float, high: float) -> None:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(low, high)

    _clip("attendance_pct", 0, 100)
    _clip("study_hours_per_day", 0, 24)
    _clip("prev_gpa", 0, 10)
    _clip("assignments_submitted", 0, 100)
    _clip("extracurricular_score", 0, 10)
    _clip("counseling_sessions", 0, 50)
    _clip("age", 10, 100)

    return df


class SentimentFeatures(BaseEstimator, TransformerMixin):
    """
    Extracts VADER sentiment scores from the 'behavioral_notes' text column.
    Output: 6 numeric features per row.
    """

    def __init__(self, text_col: str = "behavioral_notes"):
        self.text_col = text_col
        self._analyzer: Optional[SentimentIntensityAnalyzer] = None

    def fit(self, X, y=None):
        self._analyzer = SentimentIntensityAnalyzer()
        return self

    def transform(self, X):
        if self._analyzer is None:
            self._analyzer = SentimentIntensityAnalyzer()

        if isinstance(X, pd.DataFrame):
            texts = X[self.text_col].fillna("").astype(str).tolist()
        elif isinstance(X, (list, tuple, np.ndarray)):
            arr = np.asarray(X)
            if arr.ndim == 2 and arr.shape[1] == 1:
                texts = ["" if v is None else str(v) for v in arr[:, 0]]
            else:
                texts = ["" if v is None else str(v) for v in arr]
        else:
            texts = [str(X)]

        feats = np.zeros((len(texts), 6), dtype=float)
        for i, t in enumerate(texts):
            s = self._analyzer.polarity_scores(t)
            word_count = len(t.split())
            char_count = len(t)
            feats[i, :] = [
                float(s.get("neg", 0.0)),
                float(s.get("neu", 0.0)),
                float(s.get("pos", 0.0)),
                float(s.get("compound", 0.0)),
                float(word_count),
                float(char_count),
            ]
        return feats

    def get_feature_names_out(self, input_features=None):
        return np.array(
            [
                "sent_neg",
                "sent_neu",
                "sent_pos",
                "sent_compound",
                "notes_word_count",
                "notes_char_count",
            ],
            dtype=object,
        )
 