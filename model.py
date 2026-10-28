"""scikit-learn pipelines: lexical features + TF-IDF tokens -> classifier.

Both classifiers named in the report are provided:
  * "random_forest" - Random Forest (ensemble of decision trees)
  * "svm"           - Support Vector Machine (RBF kernel)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from features import FEATURE_NAMES, extract_features, extract_tokens

MODEL_NAMES = ("random_forest", "svm")


def _lexical_matrix(frame: pd.DataFrame) -> np.ndarray:
    rows = [[extract_features(u)[k] for k in FEATURE_NAMES] for u in frame["url"]]
    return np.asarray(rows, dtype=float)


def _token_text(frame: pd.DataFrame) -> list[str]:
    return [extract_tokens(u) for u in frame["url"]]


def make_classifier(name: str):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if name == "svm":
        # Calibration gives well-behaved probabilities (SVC's probability=True is deprecated).
        return CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale"), cv=3)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}.")


def build_pipeline(name: str) -> Pipeline:
    """Build the full pipeline. Input: DataFrame with a 'url' column."""
    features = ColumnTransformer([
        ("lexical", Pipeline([
            ("extract", FunctionTransformer(_lexical_matrix)),
            ("scale", StandardScaler()),
        ]), ["url"]),
        ("tokens", Pipeline([
            ("text", FunctionTransformer(_token_text)),
            ("tfidf", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                      sublinear_tf=True)),
        ]), ["url"]),
    ])
    return Pipeline([("features", features), ("clf", make_classifier(name))])
