import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import FEATURE_NAMES, extract_features, extract_tokens  # noqa: E402


def test_every_feature_present_and_numeric():
    feats = extract_features("https://www.example.com/a/b?x=1")
    assert set(feats) == set(FEATURE_NAMES)
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_ip_address_host_detected():
    assert extract_features("http://192.168.10.5/login")["has_ip_address"] == 1
    assert extract_features("https://example.com")["has_ip_address"] == 0


def test_at_symbol_and_redirect_counted():
    feats = extract_features("http://paypal.com@evil.xyz//redirect")
    assert feats["num_at_symbol"] == 1
    assert feats["num_double_slash_redirect"] == 1


def test_suspicious_words_and_brand_outside_domain():
    feats = extract_features("http://paypal-secure-login.evil.xyz/verify/account")
    assert feats["has_suspicious_word"] == 1
    assert feats["num_suspicious_words"] >= 3
    assert feats["brand_outside_domain"] == 1


def test_legitimate_brand_inside_own_domain_not_flagged():
    assert extract_features("https://www.paypal.com/home")["brand_outside_domain"] == 0


def test_subdomain_and_https_and_shortener():
    feats = extract_features("https://a.b.c.example.com/")
    assert feats["num_subdomains"] == 3
    assert feats["uses_https"] == 1
    assert extract_features("http://bit.ly/abc")["is_shortener"] == 1


def test_scheme_is_optional():
    assert extract_features("example.com/path")["path_length"] == len("/path")


def test_malformed_url_does_not_raise():
    extract_features("http://[::bad")  # must not raise


def test_tokens_are_lowercase_bag_of_words():
    assert extract_tokens("https://Secure-Login.Example.com/verify?id=42") == \
        "https secure login example com verify id 42"
