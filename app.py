"""Flask web application: 'Legitimate URL detection' front end and JSON API."""
from __future__ import annotations

import os
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from predict import DEFAULT_MODEL, load_classifier

MAX_URL_LENGTH = 2048

app = Flask(__name__)


@lru_cache(maxsize=4)
def get_classifier(name: str):
    return load_classifier(name)


def _validate(url: str) -> str | None:
    if not url:
        return "Please enter a URL."
    if len(url) > MAX_URL_LENGTH:
        return f"URL is too long (max {MAX_URL_LENGTH} characters)."
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    url, result, error = "", None, None
    if request.method == "POST":
        url = request.form.get("long_url", "").strip()
        error = _validate(url)
        if error is None:
            try:
                result = get_classifier(DEFAULT_MODEL).predict(url)
            except FileNotFoundError as exc:
                error = str(exc)
    return render_template("index.html", url=url, result=result, error=error)


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url") or request.form.get("url", "")).strip()
    error = _validate(url)
    if error:
        return jsonify(error=error), 400
    try:
        return jsonify(get_classifier(DEFAULT_MODEL).predict(url))
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 503


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
