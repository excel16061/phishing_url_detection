"""Load a trained model and classify URLs.

CLI:
    python predict.py "http://secure-login.example.xyz/verify/account"
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from features import extract_features

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("PHISHING_MODEL", "random_forest")


class UrlClassifier:
    def __init__(self, model_path: Path):
        self.pipeline = joblib.load(model_path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, url: str) -> dict:
        frame = pd.DataFrame({"url": [url]})
        proba = self.pipeline.predict_proba(frame)[0]
        p_phishing = float(proba[self.classes.index(1)])
        is_phishing = p_phishing >= 0.5
        return {
            "url": url,
            "label": "phishing" if is_phishing else "legitimate",
            "is_phishing": is_phishing,
            "phishing_probability": round(p_phishing, 4),
            "features": extract_features(url),
        }


def load_classifier(name: str = DEFAULT_MODEL,
                    model_dir: Path = BASE_DIR / "models") -> UrlClassifier:
    path = model_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python train.py` first.")
    return UrlClassifier(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a URL as legitimate or phishing")
    parser.add_argument("url")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=["random_forest", "svm"])
    args = parser.parse_args()
    result = load_classifier(args.model).predict(args.url)
    print(json.dumps(result, indent=2))
