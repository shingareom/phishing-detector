from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import requests
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher

# ==============================
# Load pipeline
# ==============================
obj = joblib.load("phishing_pipeline.pkl")

if isinstance(obj, tuple) and len(obj) == 3:
    model, le, model_columns = obj
else:
    model = obj
    le, model_columns = None, model.feature_names_in_.tolist()

explainer = shap.Explainer(model)

app = FastAPI(title="Phishing URL Detector API")

class URLRequest(BaseModel):
    url: str

# ------------------------------
# Helper functions
# ------------------------------
def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

common_brands = ["google", "facebook", "paypal", "amazon", "microsoft", "bank"]

tld_legit_prob = {
    "com": 0.95, "org": 0.9, "net": 0.85, "gov": 0.99, "edu": 0.98,
    "xyz": 0.3, "top": 0.2, "info": 0.5
}

# ------------------------------
# Feature extraction
# ------------------------------
def extract_url_features(url: str):
    features = {}

    # --- Suspicious TLDs and Keywords ---
    phishing_tlds = {"xyz", "top", "tk", "buzz", "club", "work", "gq"}
    phishing_keywords = ["login", "secure", "update", "verify", "signin", "account", "banking"]

    # --- String-based features ---
    features["URLLength"] = len(url)
    domain = url.split("//")[-1].split("/")[0]
    features["DomainLength"] = len(domain)
    features["IsDomainIP"] = int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)))
    features["TLD"] = domain.split(".")[-1] if "." in domain else "none"

    # Phishing TLD indicator
    features["SuspiciousTLD"] = int(features["TLD"].lower() in phishing_tlds)

    # Phishing keyword indicator
    features["SuspiciousKeyword"] = int(any(k in url.lower() for k in phishing_keywords))

    # Similarity to brand names
    features["URLSimilarityIndex"] = max(string_similarity(domain, b) for b in common_brands)

    # Character continuation rate
    features["CharContinuationRate"] = max((len(m.group()) for m in re.finditer(r"(.)\1+", url)), default=1) / len(url)

    # TLD legitimacy probability
    features["TLDLegitimateProb"] = tld_legit_prob.get(features["TLD"].lower(), 0.5)

    # URL character probability → letters / total
    features["URLCharProb"] = sum(c.isalpha() for c in url) / max(1, len(url))

    # Ratios & counts
    features["TLDLength"] = len(features["TLD"])
    features["NoOfSubDomain"] = domain.count(".")
    features["HasObfuscation"] = int("@" in url or "-" in url)
    features["NoOfObfuscatedChar"] = url.count("@") + url.count("-")
    features["ObfuscationRatio"] = features["NoOfObfuscatedChar"] / max(1, len(url))
    features["NoOfLettersInURL"] = sum(c.isalpha() for c in url)
    features["LetterRatioInURL"] = features["NoOfLettersInURL"] / max(1, len(url))
    features["NoOfDegitsInURL"] = sum(c.isdigit() for c in url)
    features["DegitRatioInURL"] = features["NoOfDegitsInURL"] / max(1, len(url))
    features["NoOfEqualsInURL"] = url.count("=")
    features["NoOfQMarkInURL"] = url.count("?")
    features["NoOfAmpersandInURL"] = url.count("&")
    features["NoOfOtherSpecialCharsInURL"] = len(re.findall(r"[^a-zA-Z0-9]", url))
    features["SpacialCharRatioInURL"] = features["NoOfOtherSpecialCharsInURL"] / max(1, len(url))
    features["IsHTTPS"] = int(url.startswith("https"))

    # --- HTML-based features ---
    try:
        response = requests.get(url, timeout=5)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        lines = html.splitlines()
        features["LineOfCode"] = len(lines)
        features["LargestLineLength"] = max(len(line) for line in lines) if lines else 0
        features["HasTitle"] = int(bool(soup.title))

        title_text = soup.title.string if soup.title else ""
        features["DomainTitleMatchScore"] = string_similarity(domain, title_text)
        features["URLTitleMatchScore"] = string_similarity(url, title_text)

        features["HasFavicon"] = int(bool(soup.find("link", rel="icon")))
        features["Robots"] = int("robots" in html.lower())
        features["IsResponsive"] = int("viewport" in html.lower())
        features["NoOfURLRedirect"] = html.count("http")
        features["NoOfSelfRedirect"] = html.count(url)
        features["HasDescription"] = int(bool(soup.find("meta", attrs={"name": "description"})))
        features["NoOfPopup"] = html.lower().count("popup")
        features["NoOfiFrame"] = len(soup.find_all("iframe"))
        features["HasExternalFormSubmit"] = int(any("http" in f.get("action", "") for f in soup.find_all("form")))
        features["HasSocialNet"] = int(any(s in html.lower() for s in ["facebook", "twitter", "instagram"]))
        features["HasSubmitButton"] = int(bool(soup.find("button", {"type": "submit"})))
        features["HasHiddenFields"] = int(bool(soup.find("input", {"type": "hidden"})))
        features["HasPasswordField"] = int(bool(soup.find("input", {"type": "password"})))
        features["Bank"] = int("bank" in html.lower())
        features["Pay"] = int("pay" in html.lower())
        features["Crypto"] = int("crypto" in html.lower())
        features["HasCopyrightInfo"] = int("copyright" in html.lower())
        features["NoOfImage"] = len(soup.find_all("img"))
        features["NoOfCSS"] = len(soup.find_all("link", rel="stylesheet"))
        features["NoOfJS"] = len(soup.find_all("script"))
        features["NoOfSelfRef"] = html.count(url)
        features["NoOfEmptyRef"] = html.count('href=""')
        features["NoOfExternalRef"] = html.count("http") - html.count(url)

    except Exception:
        for col in model_columns:
            if col not in features:
                features[col] = 0

    return pd.DataFrame([features])

# ------------------------------
# Prediction Endpoint
# ------------------------------
@app.post("/predict")
def predict(request: URLRequest):
    url = request.url
    try:
        X_new = extract_url_features(url)

        for col in model_columns:
            if col not in X_new.columns:
                X_new[col] = 0

        X_new = X_new[model_columns]

        if le is not None and "TLD" in X_new.columns:
            try:
                X_new["TLD"] = le.transform(X_new["TLD"])
            except ValueError:
                X_new["TLD"] = 0

        pred = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0].tolist()

        shap_values = explainer(X_new)
        shap_array = (
            shap_values.values[0][:, 1]
            if shap_values.values.ndim == 3
            else shap_values.values[0]
        )

        shap_df = pd.DataFrame({
            "Feature": X_new.columns,
            "SHAP_Value": shap_array
        }).sort_values(by="SHAP_Value", key=abs, ascending=False)

        top_features = shap_df.head(5).to_dict(orient="records")

        return {
            "url": url,
            "prediction": int(pred),
            "probabilities": prob,
            "top_features": top_features
        }

    except Exception as e:
        return {"error": str(e)}
