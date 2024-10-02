from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model from the models directory
model_path = 'models/fraud_detection_model.pkl'

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("Model loaded successfully.")
else:
    print("Model file not found. Please run the training script.")

# Create a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get transaction data from the request
    data = request.json
    
    # Convert the data to the format expected by the model (adapt based on your dataset features)
    features = np.array([data['Time'], data['Amount'], data['V1'], data['V2'], data['V3'], data['V4'], data['V5'], data['V6'], data['V7'], data['V8'], data['V9'], data['V10']])
    
    # Make a prediction
    fraud_prediction = model.predict([features])[0]
    
    # Return the result
    return jsonify({"fraud_prediction": int(fraud_prediction)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
