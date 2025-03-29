document.addEventListener("DOMContentLoaded", function () {
    const resultContainer = document.getElementById("result-container");
    const resultText = document.getElementById("result");
    const urlElement = document.getElementById("url");
    const reportButton = document.getElementById("report-false");
    const scanAgainButton = document.getElementById("scan-again");

    // Function to analyze the current URL
    function analyzeUrl() {
        chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
            const url = tabs[0].url;
            urlElement.textContent = `🔗 URL: ${url}`;
            resultText.textContent = "🔍 Analyzing...";
            resultContainer.classList.remove("phishing", "safe");

            // Make a POST request to the Flask backend
            fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ url: url }),
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Backend response:", data);  // Debug log
                if (data.error) {
                    throw new Error(data.error);
                }

                // Update UI based on prediction
                if (data.phishing) {
                    resultText.textContent = "🚨 Phishing Website Detected!";
                    resultContainer.classList.add("phishing");
                } else {
                    resultText.textContent = "✅ Safe Website";
                    resultContainer.classList.add("safe");
                }

                // Show probability score
                const probability = (data.probability * 100).toFixed(2);
                resultText.textContent += ` (${probability}% confidence)`;
            })
            .catch(error => {
                console.error("Error:", error);
                resultText.textContent = "❌ Error fetching results. Please try again.";
                resultContainer.classList.remove("phishing", "safe");
            });
        });
    }

    // Initial analysis
    analyzeUrl();

    // Handle "Scan Again" button click
    scanAgainButton.addEventListener("click", function () {
        analyzeUrl();
    });

    // Handle "Report False Result" button click
    reportButton.addEventListener("click", function () {
        alert("Thank you! Your report will be reviewed.");
    });
});