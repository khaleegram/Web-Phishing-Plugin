import re
import joblib
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import

from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load artifacts with fallback
model = load_model("phishing_detector.keras")
scaler = joblib.load("scaler.pkl")
optimal_threshold = joblib.load("optimal_threshold.pkl")

# Critical: Maintain EXACT training feature order
top_48_features = [
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
]

class FeatureExtractor:
    def __init__(self, url):
        self.url = unquote(url)
        self.parsed = urlparse(self.url)
        self.domain = self.parsed.netloc.split(':')[0]
        self.tld = self.domain.split('.')[-1] if '.' in self.domain else ''
        self.html_content = None
        self.soup = None
        
        try:
            # Bypass SSL verification and set timeout
            response = requests.get(self.url, 
                                  headers={'User-Agent': 'Mozilla/5.0'},
                                  timeout=5,
                                  verify=False,
                                  allow_redirects=True)
            self.html_content = response.text
            self.soup = BeautifulSoup(self.html_content, 'html.parser')
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self._create_fallback_features()

    def _create_fallback_features(self):
        """Create minimal features when page can't be loaded"""
        self.html_content = ""
        self.soup = BeautifulSoup("", 'html.parser')

    def _safe_divide(self, a, b):
        return a / b if b != 0 else 0

    def _char_continuation_rate(self, text):
        return sum(1 for i in range(1, len(text)) if text[i] == text[i-1]) / len(text) if len(text) > 1 else 0

    def extract_features(self):
        features = {f: 0 for f in top_48_features}
        text = self.html_content.lower()
        
        # === URL Features ===
        # [Keep all your URL feature calculations unchanged]
        
        # === HTML Features ===
        features['LineOfCode'] = len(self.html_content.split('\n'))
        features['LargestLineLength'] = max(len(line) for line in self.html_content.split('\n'))
        
        # Simplified element counting
        features['NoOfJS'] = self.html_content.count('<script')
        features['NoOfCSS'] = self.html_content.count('stylesheet')
        features['NoOfiFrame'] = self.html_content.count('<iframe')
        
        # === Security Features ===
        features['HasObfuscation'] = int(
            'javascript:' in self.url.lower() or 
            any(c in self.url for c in ['%', '\\x', '&#']) or
            'eval(' in text
        )
        
        # === Critical Fixes ===
        features['DomainTitleMatchScore'] = int(self.domain in text[:500])  # Simplified
        features['URLCharProb'] = sum(ord(c) for c in self.url) / 10000  # Normalized
        
        return [features[f] for f in top_48_features]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received request with data:", data)  # Debug log
    url = data.get('url')
 
    
    if not url:
        return jsonify({'error': 'URL required'}), 400

    try:
        extractor = FeatureExtractor(url)
        features = extractor.extract_features()
        
        # Convert to numpy array with correct shape
        scaled = scaler.transform([features])
        reshaped = scaled.reshape(1, len(top_48_features), 1)
        
        prob = model.predict(reshaped)[0][0]
        adjusted_prob = max(min(prob, 0.99), 0.01)  # Prevent extreme values
        
        return jsonify({
            'phishing': bool(adjusted_prob > optimal_threshold),
            'probability': float(adjusted_prob),
            'threshold': float(optimal_threshold)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    app.run(host='0.0.0.0', port=5000)