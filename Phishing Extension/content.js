chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "checkURL") {
        let currentURL = window.location.href;
        fetch(`http://127.0.0.1:5000/predict?url=${encodeURIComponent(currentURL)}`)
            .then(response => response.json())
            .then(data => {
                sendResponse({ phishing: data.phishing, confidence: data.confidence });
            })
            .catch(error => {
                console.error("Error:", error);
                sendResponse({ error: "Failed to analyze URL" });
            });
    }
    return true;
});
