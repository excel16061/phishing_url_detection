"""Generate a SYNTHETIC demo dataset (data/urls.csv).

The real labelled dataset used in the report is not included. This script
produces URLs that imitate common patterns (IP hosts, brand names in
subdomains, '@' tricks, login keywords, suspicious TLDs) so the full pipeline
can be run end-to-end. Accuracy on this data says nothing about real-world
performance: replace it with a real labelled corpus (e.g. PhishTank,
OpenPhish, or a Kaggle malicious-URL dataset) with columns url,label.

Usage:  python scripts/generate_sample_data.py [--n 600] [--seed 7]
"""
from __future__ import annotations

import argparse
import csv
import random
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LEGIT_HOSTS = [
    "www.wikipedia.org", "github.com", "www.python.org", "docs.python.org",
    "stackoverflow.com", "www.bbc.co.uk", "www.nytimes.com", "www.nasa.gov",
    "www.gov.uk", "www.mit.edu", "www.khanacademy.org", "www.coursera.org",
    "scikit-learn.org", "pandas.pydata.org", "www.kaggle.com", "news.ycombinator.com",
    "www.amazon.com", "www.paypal.com", "accounts.google.com", "mail.google.com",
    "www.facebook.com", "www.linkedin.com", "www.netflix.com", "www.apple.com",
    "en.wikipedia.org", "www.reddit.com", "www.bbc.com", "www.who.int",
]
LEGIT_WORDS = ["wiki", "articles", "docs", "guide", "news", "courses", "tutorial",
               "about", "products", "help", "blog", "release", "project", "search"]
PHISH_BRANDS = ["paypal", "ebay", "amazon", "apple", "microsoft", "netflix",
                "facebook", "dropbox", "wellsfargo", "chase", "icloud"]
PHISH_WORDS = ["secure", "login", "signin", "verify", "account", "update",
               "confirm", "banking", "webscr", "validate", "unlock"]
SUSPICIOUS_TLDS = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "info"]


def rand_word(rng: random.Random, low: int = 4, high: int = 9) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(low, high)))


def rand_hex(rng: random.Random, n: int = 16) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def legit_url(rng: random.Random) -> str:
    host = rng.choice(LEGIT_HOSTS)
    depth = rng.randint(0, 3)
    path = "/".join(rng.choice(LEGIT_WORDS) + (str(rng.randint(1, 2024)) if rng.random() < 0.2 else "")
                    for _ in range(depth))
    url = f"https://{host}/{path}" if path else f"https://{host}/"
    if rng.random() < 0.3:
        url += f"?q={rand_word(rng, 3, 10)}&page={rng.randint(1, 9)}"
    return url


def phish_url(rng: random.Random) -> str:
    brand = rng.choice(PHISH_BRANDS)
    word = rng.choice(PHISH_WORDS)
    tld = rng.choice(SUSPICIOUS_TLDS)
    pattern = rng.randrange(6)
    if pattern == 0:  # raw IP host
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{brand}/{word}.php?cmd=_login&session={rand_hex(rng)}"
    if pattern == 1:  # brand + keyword hyphenated under a cheap domain
        return f"http://{brand}-{word}-{rand_word(rng, 3, 6)}.{rand_word(rng, 5, 10)}.{tld}/{word}/{word}"
    if pattern == 2:  # brand used as a subdomain of an unrelated domain
        return f"http://{brand}.com.{rand_word(rng)}.{tld}/{word}/{word}-center/index.html"
    if pattern == 3:  # '@' trick hiding the real host
        return f"http://{brand}.com/{word}@{rand_word(rng)}.{tld}/{word}?id={rand_hex(rng, 24)}"
    if pattern == 4:  # redirect parameter and percent-encoding
        return (f"https://{rand_word(rng)}{rng.randint(10, 999)}.{tld}/redirect//"
                f"{word}%2F{brand}%2F{rand_word(rng)}?continue=http%3A%2F%2F{rand_word(rng)}.{tld}")
    # long multi-level path with several security words
    levels = "/".join(rng.choice(PHISH_WORDS) for _ in range(rng.randint(3, 5)))
    return f"http://{rand_word(rng, 6, 12)}-{word}.{rand_word(rng, 4, 8)}.{tld}/{levels}/{rand_hex(rng, 8)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="URLs per class")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "urls.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = {(legit_url(rng), "legitimate") for _ in range(args.n)}
    rows |= {(phish_url(rng), "phishing") for _ in range(args.n)}
    rows = sorted(rows)
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {args.out}")


if __name__ == "__main__":
    main()
