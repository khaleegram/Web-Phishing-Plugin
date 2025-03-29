import os
import re
import numpy as np
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import

from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and scaler
scaler = joblib.load('saved_model/scaler.pkl')
model = load_model('saved_model/phishing_detector.keras')

top_48_features = [
    # Original 48 features (remove the last 2 new ones)
    'URLSimilarityIndex', 'LineOfCode', 'NoOfExternalRef', 'NoOfImage', 'NoOfSelfRef',
    'NoOfJS', 'LargestLineLength', 'NoOfCSS', 'HasSocialNet', 'LetterRatioInURL',
    'HasCopyrightInfo', 'HasDescription', 'NoOfOtherSpecialCharsInURL', 'IsHTTPS',
    'SpacialCharRatioInURL', 'DomainTitleMatchScore', 'HasSubmitButton', 'TLDLegitimateProb',
    'URLLength', 'DegitRatioInURL', 'NoOfEmptyRef', 'NoOfDegitsInURL', 'IsResponsive',
    'URLTitleMatchScore', 'NoOfiFrame', 'CharContinuationRate', 'NoOfLettersInURL',
    'HasHiddenFields', 'HasFavicon', 'HasTitle', 'URLCharProb', 'DomainLength',
    'Robots', 'Pay', 'NoOfSubDomain', 'NoOfPopup', 'TLDLength', 'NoOfEqualsInURL',
    'HasExternalFormSubmit', 'NoOfQMarkInURL', 'Bank', 'HasPasswordField',
    'NoOfSelfRedirect', 'Crypto', 'HasObfuscation', 'NoOfAmpersandInURL', 'IsDomainIP', 'ObfuscationRatio'
    # Removed: 'SuspiciousPath', 'URLShortener'
]

def fetch_html(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

class EnhancedFeatureExtractor:
    SUSPICIOUS_TLDS = {'xyz', 'top', 'cfd', 'gq', 'ml', 'tk', 'men', 'work'}
    URL_SHORTENERS = {'bit.ly', 'goo.gl', 'tinyurl', 'ow.ly', 't.co', 'is.gd'}
    
    def __init__(self, url):
        self.original_url = url
        self.parsed_url = urlparse(unquote(url))
        self.domain = self.parsed_url.netloc.split(':')[0]  # Remove port
        self.tld = self.domain.split('.')[-1] if '.' in self.domain else ''
        self.html_content = fetch_html(url)
        self.soup = BeautifulSoup(self.html_content, 'html.parser') if self.html_content else None
        self.safe_html = self.html_content or ""
        self.lower_html = self.safe_html.lower()

    def extract_features(self):
        features = {feature: 0 for feature in top_48_features}

        # URL Structure Features
        features = self._extract_url_features(features)
        
        # Domain Features
        features = self._extract_domain_features(features)
        
        # HTML Content Features
        features = self._extract_html_features(features)
        
        # Security Features
        features = self._extract_security_features(features)
        
        # Behavioral Features
        features = self._extract_behavioral_features(features)
        
        # New Features
        features['SuspiciousPath'] = self._has_suspicious_path()
        features['URLShortener'] = self._is_url_shortener()

        return [features[feature] for feature in top_48_features]

    def _extract_url_features(self, features):
        url = self.original_url.lower()
        path = self.parsed_url.path.lower()
        
        # Basic URL features
        features['IsHTTPS'] = 1 if self.parsed_url.scheme == 'https' else 0
        features['URLLength'] = len(self.original_url)
        features['NoOfDegitsInURL'] = sum(c.isdigit() for c in self.original_url)
        features['DegitRatioInURL'] = self._safe_divide(features['NoOfDegitsInURL'], features['URLLength'])
        features['NoOfLettersInURL'] = sum(c.isalpha() for c in self.original_url)
        features['LetterRatioInURL'] = self._safe_divide(features['NoOfLettersInURL'], features['URLLength'])
        
        # Special characters
        special_chars = set('!@#$%^&*()_+{}[]|\\;\'",<>?/~`')
        features['NoOfOtherSpecialCharsInURL'] = sum(1 for c in self.original_url if c in special_chars)
        features['SpacialCharRatioInURL'] = self._safe_divide(features['NoOfOtherSpecialCharsInURL'], features['URLLength'])
        
        # Query parameters
        features['NoOfQMarkInURL'] = self.original_url.count('?')
        features['NoOfEqualsInURL'] = self.original_url.count('=')
        features['NoOfAmpersandInURL'] = self.original_url.count('&')
        
        # Suspicious keywords
        features['Pay'] = 1 if 'pay' in url or 'pay' in self.lower_html else 0
        features['Bank'] = 1 if 'bank' in url or 'bank' in self.lower_html else 0
        features['Crypto'] = 1 if 'crypto' in url or 'crypto' in self.lower_html else 0
        
        return features

    def _extract_domain_features(self, features):
        features['DomainLength'] = len(self.domain)
        features['TLDLength'] = len(self.tld)
        features['TLDLegitimateProb'] = 1 if self.tld in {'com', 'org', 'net', 'edu', 'gov'} else 0
        features['IsDomainIP'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', self.domain) else 0
        features['NoOfSubDomain'] = len(self.domain.split('.')) - 2 if len(self.domain.split('.')) > 2 else 0
        features['URLSimilarityIndex'] = self._safe_divide(len(self.domain), features['URLLength'])
        return features

    def _extract_html_features(self, features):
        if not self.soup:
            return features

        # Basic HTML structure
        lines = self.safe_html.split('\n')
        features['LineOfCode'] = len(lines)
        features['LargestLineLength'] = max(len(line) for line in lines) if lines else 0
        
        # Resource counting
        features['NoOfImage'] = len(self.soup.find_all('img'))
        features['NoOfJS'] = len(self.soup.find_all('script'))
        features['NoOfCSS'] = len(self.soup.find_all('link', rel='stylesheet'))
        features['NoOfiFrame'] = len(self.soup.find_all('iframe'))
        
        # Form elements
        features['HasPasswordField'] = 1 if self.soup.find('input', {'type': 'password'}) else 0
        features['HasSubmitButton'] = 1 if self.soup.find('input', {'type': 'submit'}) or self.soup.find('button', {'type': 'submit'}) else 0
        features['HasHiddenFields'] = len(self.soup.find_all('input', {'type': 'hidden'}))
        
        # Meta tags
        features['HasFavicon'] = 1 if self.soup.find('link', rel='icon') else 0
        features['HasTitle'] = 1 if self.soup.title else 0
        features['HasDescription'] = 1 if self.soup.find('meta', attrs={'name': 'description'}) else 0
        features['IsResponsive'] = 1 if self.soup.find('meta', attrs={'name': 'viewport'}) else 0
        
        # References
        features['NoOfExternalRef'] = sum(1 for tag in self.soup.find_all(['img', 'script', 'link']) 
            if self._is_external_reference(tag))
        features['NoOfSelfRef'] = sum(1 for tag in self.soup.find_all(['img', 'script', 'link']) 
            if self._is_self_reference(tag))
        
        return features

    def _extract_security_features(self, features):
        features['HasObfuscation'] = 1 if 'eval(' in self.safe_html else 0
        features['ObfuscationRatio'] = features['HasObfuscation']
        features['HasCopyrightInfo'] = 1 if 'copyright' in self.lower_html else 0
        features['HasSocialNet'] = 1 if any(social in self.lower_html for social in {'facebook', 'twitter', 'linkedin'}) else 0
        return features

    def _extract_behavioral_features(self, features):
        features['CharContinuationRate'] = self._calculate_char_continuation()
        return features

    def _calculate_char_continuation(self):
        if len(self.original_url) < 2:
            return 0
        transitions = sum(1 for i in range(1, len(self.original_url)) 
            if self.original_url[i] == self.original_url[i-1])
        return transitions / len(self.original_url)

    def _has_suspicious_path(self):
        suspicious_keywords = {'login', 'signin', 'verify', 'account', 'secure'}
        path = self.parsed_url.path.lower()
        return 1 if any(kw in path for kw in suspicious_keywords) else 0

    def _is_url_shortener(self):
        domain = self.domain.lower()
        return 1 if any(shortener in domain for shortener in self.URL_SHORTENERS) else 0

    def _is_external_reference(self, tag):
        url = tag.get('src') or tag.get('href') or ''
        parsed = urlparse(url)
        return parsed.netloc not in ['', self.domain]

    def _is_self_reference(self, tag):
        url = tag.get('src') or tag.get('href') or ''
        parsed = urlparse(url)
        return parsed.netloc == self.domain

    def _safe_divide(self, numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        extractor = EnhancedFeatureExtractor(url)
        features = extractor.extract_features()
        
        # Debug: Print feature values
        print("\nExtracted Features:")
        for name, value in zip(top_48_features, features):
            print(f"{name}: {value}")
        
        scaled_features = scaler.transform([features])
        reshaped_features = scaled_features.reshape((1, 48, 1))
        prediction = model.predict(reshaped_features)
        
        # Adjust threshold based on observed behavior
        adjusted_threshold = 0.3
        is_phishing = prediction[0][0] > adjusted_threshold
        
        return jsonify({
            'phishing': bool(is_phishing),
            'probability': float(prediction[0][0]),
            'threshold': adjusted_threshold
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)