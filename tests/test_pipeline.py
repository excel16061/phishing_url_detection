import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import MODEL_NAMES, build_pipeline  # noqa: E402
from train import load_dataset  # noqa: E402


def _toy_frame():
    legit = [f"https://www.site{i}.com/articles/{i}" for i in range(30)]
    phish = [f"http://secure-login-paypal.evil{i}.xyz/verify/account?id={i}" for i in range(30)]
    urls = legit + phish
    labels = [0] * 30 + [1] * 30
    return pd.DataFrame({"url": urls}), pd.Series(labels)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_learns_toy_problem(name):
    X, y = _toy_frame()
    pipe = build_pipeline(name).fit(X, y)
    probe = pd.DataFrame({"url": ["http://secure-login-paypal.newevil.xyz/verify/account",
                                  "https://www.wikipedia.org/wiki/Python"]})
    preds = pipe.predict(probe)
    assert list(preds) == [1, 0]


def test_load_dataset_maps_labels(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("url,label\nhttps://a.com,legitimate\nhttp://x.xyz/login,phishing\n"
                   "https://a.com,legitimate\n")
    X, y = load_dataset(csv)
    assert len(X) == 2  # duplicate removed
    assert sorted(y.tolist()) == [0, 1]


def test_load_dataset_rejects_unknown_label(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("url,label\nhttps://a.com,maybe\n")
    with pytest.raises(ValueError):
        load_dataset(csv)
