from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import os

app = Flask(
    __name__,
    static_url_path='',
    static_folder='static',
    template_folder='templates'
)
CORS(app)

# Load the trained model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'parkinson_model.pkl')  # Your model filename
    model = joblib.load(model_path)
    print("Model loaded successfully:", type(model))
except Exception as e:
    print("Error loading model:", str(e))
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/letsget.html')
def letsget():
    return render_template('letsget.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data received"}), 400

        # Required keys from frontend
        required_keys = ["Name", "age", "Eye blinking", "handshaking", "MDVP:Fo(Hz)"]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            return jsonify({"error": f"Missing required features: {', '.join(missing_keys)}"}), 400

        # Extract and preprocess input
        name = data["Name"].strip()
        age = float(data["age"])
        eye_blinking = int(data["Eye blinking"])
        handshaking = int(data["handshaking"])
        fo_hz = float(data["MDVP:Fo(Hz)"])

        # Validations
        if age >= 60:
            return jsonify({"error": "This is early Parkinson detection, please enter age below 60."}), 400
        if eye_blinking not in [0, 1]:
            return jsonify({"error": "Eye blinking must be 0 or 1."}), 400
        if handshaking not in [0, 1]:
            return jsonify({"error": "Handshaking must be 0 or 1."}), 400

        # Internally compute the 5th feature: symptom_score
        symptom_score = eye_blinking + handshaking

        # Prepare the input array (5 features)
        features = np.array([[age, eye_blinking, handshaking, fo_hz, symptom_score]])

        # Perform prediction
        if model is None:
            return jsonify({"error": "Model not loaded properly"}), 500

        detection_result = model.predict(features)[0]

        # Response message
        if detection_result == 1:
            message = f"{name}, may this be costly information! You are detected with Parkinson’s disease."
        else:
            message = f"Congratulations {name}, on a great escape from Parkinson’s disease."

        return jsonify({"detection_result": message, "success": True})

    except Exception as e:
        print("Error occurred:", str(e))
        print(traceback.format_exc())
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
