from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from features import extract_features

app = Flask(__name__)

model         = joblib.load('models/best_model.pkl')
tfidf         = joblib.load('models/tfidf.pkl')
feature_names = joblib.load('models/feature_names.pkl')

with open('models/model_info.json', 'r') as f:
    model_info = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify(model_info)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url

    features = extract_features(url)
    if features is None:
        return jsonify({'error': 'Could not extract features from URL'}), 400

    # Handcrafted numeric features
    hand_df     = pd.DataFrame([features])[feature_names]
    hand_sparse = sp.csr_matrix(hand_df.values)

    # TF-IDF on raw URL
    url_tfidf = tfidf.transform([url])

    # Combine exactly as during training
    X_combined = sp.hstack([hand_sparse, url_tfidf])

    prediction  = model.predict(X_combined)[0]
    probability = model.predict_proba(X_combined)[0]
    risk_score  = round(float(probability[1]) * 100, 2)

    return jsonify({
        'url':        url,
        'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
        'risk_score': risk_score,
        'safe':       bool(prediction == 0),
    })

if __name__ == '__main__':
    app.run(debug=True)
