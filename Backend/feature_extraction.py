import re
import datetime
import numpy as np
import whois
from urllib.parse import urlparse
import tldextract
import math

def calculate_entropy(string):
    """Calculates Shannon entropy for a given string."""
    if not string:
        return 0
    prob = [float(string.count(c)) / len(string) for c in set(string)]
    return -sum([p * math.log2(p) for p in prob])

def extract_features(url):
    """Extracts numerical features from a URL for phishing detection."""
    try:
        parsed_url = urlparse(url)
        domain_info = tldextract.extract(url)
        domain = domain_info.domain + "." + domain_info.suffix if domain_info.suffix else domain_info.domain

        # Basic URL Features
        url_length = len(url)
        has_ip = 1 if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", domain) else 0
        num_dots = url.count('.')
        num_hyphens = url.count('-')
        num_slashes = url.count('/')
        num_digits = sum(c.isdigit() for c in url)
        num_special_chars = sum(c in "!@#$%^&*()" for c in url)
        subdomain_count = domain_info.subdomain.count('.') + 1 if domain_info.subdomain else 0

        # URL Path Features
        path_length = len(parsed_url.path)
        has_query_string = 1 if parsed_url.query else 0
        has_fragments = 1 if parsed_url.fragment else 0

        # WHOIS Features
        try:
            whois_data = whois.whois(domain)
            creation_date = whois_data.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            age_days = (datetime.datetime.now() - creation_date).days if creation_date else 0
        except Exception:
            age_days = 0  # Default to 0 instead of -1

        # Security Features
        has_https = 1 if url.startswith("https") else 0

        # Keyword Presence
        phishing_keywords = ["secure", "login", "bank", "verify", "account", "update", "confirm"]
        keyword_presence = sum(1 for kw in phishing_keywords if kw in url.lower())

        # URL Shortener Check
        shortener_pattern = r"\b[a-zA-Z0-9-]+\.ly\b|bit\.ly|tinyurl|t\.co|goo\.gl|is\.gd|ow\.ly"
        is_shortened = 1 if re.search(shortener_pattern, domain) else 0

        # Entropy Measures
        domain_entropy = calculate_entropy(domain)
        path_entropy = calculate_entropy(parsed_url.path)

        # Feature Vector
        features = np.array([
            url_length, has_ip, num_dots, num_hyphens, num_slashes, num_digits, num_special_chars, subdomain_count,
            path_length, has_query_string, has_fragments, age_days, has_https, keyword_presence, is_shortened,
            domain_entropy, path_entropy
        ])

        # Ensure Feature Vector Matches Expected Shape (48 Features)
        if len(features) < 48:
            features = np.pad(features, (0, 48 - len(features)), 'constant')
        elif len(features) > 48:
            features = features[:48]  # Trim excess

        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(48)  # Return empty feature vector of correct size
