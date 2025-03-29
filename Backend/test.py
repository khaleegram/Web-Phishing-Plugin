import requests

# Flask API URL (change if needed)
API_URL = "http://127.0.0.1:5000/predict_url"

# List of URLs to test
TEST_URLS = [
    "https://google.com",
    "https://bank.com",
    "http://login.verify-bank.com",  # Should be flagged as phishing
    "http://paypal-security-alert.xyz",  # Should be flagged as phishing
    "https://secure.amazon.com",  # Should be safe
    "http://malicious-site.ru",  # Should be phishing
]

def test_url(url):
    """Send a POST request to Flask API and print the response."""
    payload = {"url": url}
    response = requests.post(API_URL, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"🔗 URL: {url}")
        print(f"   🛑 Is Phishing: {result['is_phishing']}")
        print(f"   🎯 Confidence: {result['confidence']}%\n")
    else:
        print(f"❌ Failed to get response for {url}")

if __name__ == "__main__":
    print("🔎 Running URL Tests...\n")
    for url in TEST_URLS:
        test_url(url)
    print("✅ Done!")
