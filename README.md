## 🛡️ PrivacyGuard Pro: Hybrid APK Security Scanner

An intelligent, context-aware security scanner for Android APKs that combines deep static analysis with an AI-powered malware detection engine.

## About The Project
PrivacyGuard Pro is a sophisticated Mobile Application Security Testing (MAST) tool designed to uncover security vulnerabilities and privacy risks in Android applications. Unlike traditional scanners that rely on simple pattern matching, this tool employs a hybrid analysis approach:

Advanced Static Analysis (SAST): It performs a deep dive into the application's code and manifest without running it. This engine uses contextual analysis to understand why a permission is requested or how an endpoint is used, significantly reducing false positives and providing actionable insights.

AI-Powered Malware Detection: It uses a machine learning model to classify applications as benign, suspicious, or malicious based on patterns learned from thousands of samples. This allows it to detect threats that might not be caught by rule-based systems.

This project was built to provide developers, security researchers, and enthusiasts with a powerful, easy-to-use tool to quickly assess the security posture of an Android application through a modern and intuitive web interface.

## Key Features
Contextual Permission Analysis: Differentiates between permissions that are justified by API calls and those that are suspiciously requested but never used.

Vulnerable Library Detection (SCA): Scans for bundled third-party libraries with known Common Vulnerabilities and Exposures (CVEs).

Intelligent Endpoint Analysis: Filters out safe, local http:// traffic and analyzes the context of public endpoints to flag those used near sensitive keywords (e.g., "login", "password").

Hardcoded Secret Detection: Scans for embedded API keys, tokens, and other sensitive strings.

Insecure Data Storage Detection: Identifies common patterns of insecurely saved user data.

AI Malware Verdict: Provides a probabilistic score and a clear verdict (Benign, Suspicious, Malicious) on the likelihood of the app being malware.

Modern Web Dashboard: A sleek and interactive interface built with Streamlit for easy APK uploading and clear visualization of results.

## Installation & Setup
Follow these steps to get your local copy of PrivacyGuard Pro up and running.

### Prerequisites
Python 3.9+

Android Studio (Optional, but recommended for its powerful emulator and SDK tools)

### Step-by-Step Setup
Clone the repository:

Bash

git clone https://github.com/your_username/privacyguard-pro.git
cd privacyguard-pro
Create and activate a Python virtual environment:

Bash

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
Install the required dependencies:

Bash

pip install streamlit androguard scikit-learn numpy requests pandas altair
Set up the Machine Learning Model:
The AI scanner requires a model and feature file. Run the included setup script once to generate these locally.

Bash

python setup_ml_model.py
This will create malware_model.joblib and model_features.json in your project directory.

## Usage
Once the installation is complete, you can launch the web dashboard with a single command.

Make sure you are in the project's root directory and your virtual environment is active.

Run the following command in your terminal:

Bash

streamlit run dashboard.py
Your web browser will automatically open a new tab with the PrivacyGuard Pro interface.

Simply drag and drop an .apk file onto the uploader to begin the analysis.
