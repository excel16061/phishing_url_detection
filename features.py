"""Lexical feature extraction for phishing URL detection.

The classifier never fetches the web page. Every feature is computed from the
URL string alone, which removes run-time latency and avoids exposing the
user's browser to page content (see Chapter 1 / Proposed System of the report).
"""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

# Security-sensitive tokens that frequently appear in phishing URLs (Chapter 2).
SUSPICIOUS_WORDS = (
    "confirm", "account", "banking", "secure", "ebayisapi",
    "webscr", "login", "signin", "verify", "update", "password",
)

# Well-known URL shortening services; they hide the real destination.
SHORTENERS = frozenset({
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly",
    "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
})

# Brand names that attackers place in URLs to look trustworthy.
BRANDS = (
    "paypal", "ebay", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "dropbox", "linkedin", "icloud", "chase", "wellsfargo",
)

# Delimiters used to split a URL into "bag-of-words" tokens (Chapter 2).
TOKEN_SPLIT = re.compile(r"[/?.=\-_&@:#%+~]+")

# Ordered list of numeric features produced by extract_features().
FEATURE_NAMES = (
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_dots_in_host",
    "num_hyphens",
    "num_at_symbol",
    "num_double_slash_redirect",
    "num_subdomains",
    "has_ip_address",
    "uses_https",
    "is_shortener",
    "num_suspicious_words",
    "has_suspicious_word",
    "brand_outside_domain",
    "num_digits",
    "digit_ratio",
    "num_special_chars",
    "num_query_params",
    "path_depth",
    "num_tokens",
    "longest_token",
    "has_percent_encoding",
    "tld_length",
)


def _split_url(raw: str):
    """Return (normalised_url, SplitResult, lowercase hostname)."""
    url = raw.strip()
    if "://" not in url:
        url = "http://" + url  # urlsplit needs a scheme to find the host
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:  # malformed netloc, e.g. unmatched bracket
        parts = urlsplit("http://invalid")
        host = ""
    return url, parts, host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_features(url: str) -> dict[str, float]:
    """Compute the lexical feature dictionary for one URL."""
    full, parts, host = _split_url(url)
    lower = full.lower()
    path = parts.path or ""
    query = parts.query or ""

    host_no_www = host[4:] if host.startswith("www.") else host
    labels = [label for label in host_no_www.split(".") if label]
    registered = labels[-2] if len(labels) >= 2 else host_no_www
    tld = labels[-1] if labels else ""

    digits = sum(ch.isdigit() for ch in full)
    tokens = [t for t in TOKEN_SPLIT.split(lower) if t]

    return {
        "url_length": len(full),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots_in_host": host.count("."),
        "num_hyphens": full.count("-"),
        "num_at_symbol": full.count("@"),
        "num_double_slash_redirect": (path + query).count("//"),
        "num_subdomains": max(len(labels) - 2, 0),
        "has_ip_address": int(_is_ip(host)),
        "uses_https": int(parts.scheme == "https"),
        "is_shortener": int(host_no_www in SHORTENERS),
        "num_suspicious_words": sum(lower.count(w) for w in SUSPICIOUS_WORDS),
        "has_suspicious_word": int(any(w in lower for w in SUSPICIOUS_WORDS)),
        "brand_outside_domain": int(
            any(b in lower and b not in registered for b in BRANDS)
        ),
        "num_digits": digits,
        "digit_ratio": round(digits / len(full), 4) if full else 0.0,
        "num_special_chars": sum(full.count(c) for c in "?=&%~+#"),
        "num_query_params": (query.count("&") + 1) if query else 0,
        "path_depth": path.count("/"),
        "num_tokens": len(tokens),
        "longest_token": max((len(t) for t in tokens), default=0),
        "has_percent_encoding": int("%" in full),
        "tld_length": len(tld),
    }


def extract_tokens(url: str) -> str:
    """Return the URL as a space-separated bag of lowercase tokens.

    This text is fed to a TF-IDF vectoriser, which is the NLP part of the
    pipeline: words such as 'login', 'secure' or a brand name get weights.
    """
    _, _, _ = _split_url(url)  # validates/normalises input
    return " ".join(t for t in TOKEN_SPLIT.split(url.strip().lower()) if t)
