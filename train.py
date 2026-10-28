"""Train and evaluate the phishing URL classifiers.

Usage:
    python train.py                       # uses data/urls.csv
    python train.py --data my_urls.csv    # any CSV with columns: url,label

Labels: legitimate/benign -> 0 ; phishing/malicious/malware/spam -> 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

from model import MODEL_NAMES, build_pipeline

BASE_DIR = Path(__file__).resolve().parent
LABEL_MAP = {
    "legitimate": 0, "benign": 0, "good": 0, "0": 0,
    "phishing": 1, "malicious": 1, "malware": 1, "spam": 1, "bad": 1, "1": 1,
}


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, dtype=str).dropna(subset=["url", "label"])
    df["url"] = df["url"].str.strip()
    df["label"] = df["label"].str.strip().str.lower()
    df = df.drop_duplicates(subset="url")
    unknown = set(df["label"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Unknown label values in {path}: {sorted(unknown)}")
    y = df["label"].map(LABEL_MAP).astype(int)
    return df[["url"]], y


def evaluate(pipeline, X_test, y_test) -> dict:
    pred = pipeline.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(
            y_test, pred, target_names=["legitimate", "phishing"],
            zero_division=0, output_dict=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "urls.csv")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"Loaded {len(X)} unique URLs from {args.data}")
    print(f"  legitimate: {(y == 0).sum()}   phishing: {(y == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for name in MODEL_NAMES:
        print(f"\nTraining {name} ...")
        pipeline = build_pipeline(name)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        all_metrics[name] = metrics
        joblib.dump(pipeline, args.out / f"{name}.joblib")
        print(f"  accuracy={metrics['accuracy']}  precision={metrics['precision']}  "
              f"recall={metrics['recall']}  f1={metrics['f1']}")
        print(f"  confusion matrix [[TN FP] [FN TP]]: {metrics['confusion_matrix']}")

    (args.out / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(f"\nModels and metrics saved to {args.out}")


if __name__ == "__main__":
    main()
