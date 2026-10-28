# Phishing URL Detection using NLP and Machine Learning

A machine-learning system that classifies a web address as **legitimate** or
**phishing/malicious** using only the URL text. The site is never fetched, so
classification has no network latency and exposes the user to no page content.

This project implements the design in the major project report
*"Detecting Phishing Attacks Using Natural Language Processing and Machine
Learning"* (Waddepally Sairam, Syed Omer Ali, Zahoor Mohammed; MITS Hyderabad,
2021-22).

## How it works

| Stage | Module | What it does |
|---|---|---|
| Lexical features | `features.py` | 24 numeric features: URL/host/path length, dots, hyphens, `@`, `//` redirects, subdomain count, raw-IP host, HTTPS, URL shorteners, suspicious words (`confirm`, `account`, `login`, `webscr`, ...), brand names outside the registered domain, digit ratio, query parameters, token statistics |
| NLP tokens | `features.py`, `model.py` | URL split into a bag of words on `/ ? . = - _ & @ :` and related delimiters; TF-IDF weights unigrams and bigrams |
| Models | `model.py` | Random Forest (200 trees) and RBF-kernel SVM (calibrated probabilities), both in scikit-learn pipelines |
| Training & evaluation | `train.py` | Stratified 80/20 split, accuracy/precision/recall/F1, confusion matrices saved to `models/metrics.json` |
| Inference | `predict.py` | Loads a saved model; returns label, phishing probability and features |
| Web app | `app.py`, `templates/`, `static/` | "Legitimate URL detection" page plus `POST /api/predict` JSON endpoint |

The report's web form asked for twelve hand-computed fields. This version
computes them from the single URL you paste, which the report's own
"Proposed System" section describes.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_sample_data.py   # optional: rebuild the demo CSV
python train.py                          # trains and saves models/*.joblib
python predict.py "http://secure-login.example.xyz/verify/account"
python app.py                            # open http://127.0.0.1:5000
pytest                                   # run the test suite
```

Use the SVM instead of the default Random Forest with
`PHISHING_MODEL=svm python app.py`, or `python predict.py URL --model svm`.

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"url": "http://1.2.3.4/paypal/login"}'
```

Response:

```json
{
  "url": "http://1.2.3.4/paypal/login",
  "label": "phishing",
  "is_phishing": true,
  "phishing_probability": 0.97,
  "features": { "has_ip_address": 1, "has_suspicious_word": 1, "...": "..." }
}
```

## Using own dataset

The included `data/urls.csv` is **synthetic**. The generator in
`scripts/generate_sample_data.py` creates URLs that imitate common phishing
patterns, so the models separate them almost perfectly. Those scores say
nothing about real-world performance.

For a meaningful evaluation, supply a labelled corpus of real URLs (for
example, PhishTank, OpenPhish, or a public Kaggle malicious-URL dataset) as a
CSV with two columns:

```csv
url,label
https://www.wikipedia.org/wiki/Python,legitimate
http://secure-login.example.xyz/verify/account,phishing
```

Accepted labels: `legitimate`, `benign`, `good`, `0` and `phishing`,
`malicious`, `malware`, `spam`, `bad`, `1`. Then run:

```bash
python train.py --data path/to/your_urls.csv
```

Check the licence and terms of any dataset you use, and do not publish raw
phishing URLs.

##  layout

```
phishing_url_detection/
├── app.py                  # Flask web app and JSON API
├── features.py             # lexical feature + token extraction
├── model.py                # scikit-learn pipelines (RF, SVM)
├── train.py                # training, evaluation, model export
├── predict.py              # inference (CLI and library)
├── data/urls.csv           # synthetic demo dataset
├── models/                 # trained pipelines and metrics.json
├── scripts/generate_sample_data.py
├── templates/index.html
├── static/css/style.css
├── tests/                  # pytest suite
└── requirements.txt
```

## Limitations

- **Binary output.** The report's abstract mentions three classes (benign,
  spam, malware), but its outputs and interface report only legitimate versus
  attack. This version is binary. Adding classes means changing `LABEL_MAP`
  in `train.py`, the labels in `predict.py`, and the template.
- **No host-reputation features.** The report used Alexa traffic rank, WHOIS
  domain age, and DNS records. Alexa's ranking service has been retired, and
  the others need live lookups, so they are not included. They would add
  accuracy if you add a provider.
- **URL-only.** Pages that are phishing but use a clean-looking URL can evade
  detection. The report calls this out as a deliberate trade-off for speed and
  safety.



