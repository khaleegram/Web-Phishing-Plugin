import os
import numpy as np
import pandas as pd
import joblib
import requests
import tensorflow as tf
from urllib.parse import urlparse
from flask import Flask, request, jsonify

# Load the trained model and scaler
model = tf.keras.models.load_model('saved_model/phishing_detector.keras')
scaler = joblib.load("saved_model/scaler.pkl")

# Define selected features
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

def extract_features(url):
    parsed_url = urlparse(url)
    features = {}
    
    features['URLLength'] = len(url)
    features['NoOfSubDomain'] = parsed_url.netloc.count('.')
    features['IsHTTPS'] = 1 if parsed_url.scheme == 'https' else 0
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / len(url)
    features['LetterRatioInURL'] = sum(c.isalpha() for c in url) / len(url)
    features['SpacialCharRatioInURL'] = sum(not c.isalnum() for c in url) / len(url)
    features['DomainLength'] = len(parsed_url.netloc)
    features['IsDomainIP'] = 1 if parsed_url.netloc.replace('.', '').isdigit() else 0
    
    # Additional features extracted from page content
    try:
        response = requests.get(url, timeout=5)
        html_content = response.text.lower()
        features['LineOfCode'] = html_content.count('\n')
        features['NoOfImage'] = html_content.count('<img')
        features['NoOfJS'] = html_content.count('<script')
        features['NoOfCSS'] = html_content.count('<style')
        features['HasSocialNet'] = 1 if any(social in html_content for social in ['facebook', 'twitter', 'instagram']) else 0
        features['HasCopyrightInfo'] = 1 if 'copyright' in html_content else 0
        features['HasDescription'] = 1 if '<meta name="description"' in html_content else 0
        features['HasSubmitButton'] = 1 if '<input type="submit"' in html_content else 0
        features['HasFavicon'] = 1 if '<link rel="shortcut icon"' in html_content else 0
        features['HasTitle'] = 1 if '<title>' in html_content else 0
        features['HasPasswordField'] = 1 if 'type="password"' in html_content else 0
        features['NoOfiFrame'] = html_content.count('<iframe')
        features['NoOfPopup'] = html_content.count('window.open')
        features['NoOfSelfRef'] = html_content.count('self.location')
        features['NoOfExternalRef'] = html_content.count('http')
        features['NoOfEmptyRef'] = html_content.count('href=""')
        features['NoOfSelfRedirect'] = html_content.count('window.location')
        features['HasObfuscation'] = 1 if 'eval(' in html_content or 'unescape(' in html_content else 0
        features['ObfuscationRatio'] = (html_content.count('eval(') + html_content.count('unescape(')) / len(html_content)
    except:
        for feature in top_48_features:
            if feature not in features:
                features[feature] = 0
    
    return [features[feature] for feature in top_48_features]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    url = data.get("url")
    
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    features = extract_features(url)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0][0]
    
    return jsonify({"url": url, "phishing_probability": float(prediction), "is_phishing": prediction > 0.5})

if __name__ == '__main__':
    app.run(debug=True)
