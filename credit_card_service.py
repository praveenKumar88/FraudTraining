from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model
model = joblib.load('fraud_detection_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Create prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get transaction data from request
    data = request.json
    features = np.array([data['Time'], data['Amount'], data['V1'], data['V2'], data['V3'], ..., data['V28']])

    # Predict fraud probability
    fraud_probability = model.predict_proba([features])[0][1]

    return jsonify({"fraud_probability": fraud_probability, "is_fraud": fraud_probability > 0.5})

if __name__ == '__main__':
    app.run(port=5000, debug=True)