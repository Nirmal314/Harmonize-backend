from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from flask import Flask, request, jsonify
import numpy as np
import time

app = Flask(__name__)

import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

model = load_model('model.h5')
scaler = joblib.load('scaler.joblib')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()  
    
    data = request.json

    features = np.array([[item['danceability'], item['energy'], item['acousticness'], item['valence'], item['tempo']] for item in data])
    tracks = [item['track'] for item in data] 
    instrumentalness_values = np.array([item['instrumentalness'] for item in data])  

    normalized_features = scaler.transform(features)

    predictions = model.predict(normalized_features)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)

    response = []
    for i in range(len(tracks)):
        if instrumentalness_values[i] > 0.6:
            category = "instrumental"
        else:
            category = predicted_labels[i]

        tracks[i]['category'] = category

        response.append({
            'track': tracks[i],
        })

    print("Execution time: ", time.time() - start_time, "s")
    return jsonify(response)


if __name__ == '__main__':
    app.run()
