from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Suppress Flask's default logs
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load the model, scaler, and label encoder
model = load_model('model.h5')
scaler = joblib.load('scaler.joblib')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features from request data
    features = np.array([[data['danceability'], data['energy'], data['acousticness'], data['valence'], data['tempo']]])

    # Normalize the features using the pre-fitted scaler
    normalized_features = scaler.transform(features)

    # Make predictions
    prediction = model.predict(normalized_features)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform([predicted_class])

    # Minimal output in the terminal
    print(f"Predicted label: {predicted_label[0]}")
    
    return jsonify({'predicted_category': predicted_label[0]})

if __name__ == '__main__':
    app.run()
