import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import urllib.request

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# URL for downloading the dataset in real time (replace with actual URL)
dataset_url = "https://your-dataset-url.com/creditcard.csv"
dataset_path = "creditcard.csv"

# Download dataset if not already downloaded
if not os.path.exists(dataset_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_path)
    print("Download complete!")

# Load dataset
df = pd.read_csv(dataset_path)

# Define features and label
X = df.drop(columns=['Class'])  # Assuming 'Class' is the label (0 = not fraud, 1 = fraud)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Save the model to the models directory
joblib.dump(model, 'models/fraud_detection_model.pkl')
print("Model saved to models/fraud_detection_model.pkl")
