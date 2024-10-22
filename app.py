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

    # Extract features and trackIds from request data
    features = np.array([[item['danceability'], item['energy'], item['acousticness'], item['valence'], item['tempo']] for item in data])
    track_ids = [item['trackId'] for item in data]  # Extract trackIds

    # Normalize the features using the pre-fitted scaler
    normalized_features = scaler.transform(features)

    # Make predictions
    predictions = model.predict(normalized_features)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    # Prepare the response with trackId and predicted category
    response = []
    for i, label in enumerate(predicted_labels):
        response.append({
            'trackId': track_ids[i],  # Include trackId
            'predicted_category': label,
        })

    return jsonify(response)


if __name__ == '__main__':
    app.run()
