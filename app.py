import os
import joblib
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- Configuration and Loading ---
# Note: For production use, files should be securely loaded and paths checked.

MODEL_PATH = 'csat_model.h5'
PREPROCESSOR_PATH = 'preprocessor.pkl'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

# Initialize variables to hold model components
model = None
preprocessor = None
label_encoder = None

# Mock-up Categories (Inferred from preprocessor structure for the HTML interface)
# These MUST align with the data your preprocessor was trained on.
CATEGORICAL_CHOICES = {
    'channel_name': ['Phone', 'Chat', 'Email', 'Social Media'],
    'category': ['Technical Issue', 'Billing Inquiry', 'Order Status', 'General Feedback'],
    'subcategory': ['Login Failure', 'Payment Error', 'Shipping Delay', 'Product Complaint'],
    'agent_name': ['Emily Chen', 'John Smith', 'Olivia Tan', 'Michael Lee'],
    'supervisor': ['David Johnson', 'Maria Lee', 'Sarah Chen'],
    'manager': ['Robert Kim', 'Jessica Ng', 'Alex Brown'],
    'tenurebucket': ['0-1 Yr', '1-3 Yr', '3-5 Yr', '5+ Yr'],
    'agentshift': ['Morning', 'Afternoon', 'Night'],
    'handling_time_bucket': ['Low', 'Medium', 'High']
}

def load_all_artifacts():
    """Loads the model, preprocessor, and label encoder."""
    global model, preprocessor, label_encoder

    try:
        # 1. Load Keras Model
        model = load_model(MODEL_PATH)
        app.logger.info(f"Successfully loaded model from {MODEL_PATH}")

        # 2. Load Preprocessor (e.g., ColumnTransformer)
        with open(PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = joblib.load(f)
        app.logger.info(f"Successfully loaded preprocessor from {PREPROCESSOR_PATH}")

        # 3. Load Label Encoder (for output decoding)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = joblib.load(f)
        app.logger.info(f"Successfully loaded label encoder from {LABEL_ENCODER_PATH}")

    except Exception as e:
        app.logger.error(f"Error loading model artifacts: {e}")
        # In a real app, this should prevent the server from starting or mark it unhealthy
        return False
    return True

# Load artifacts immediately when the application starts
load_all_artifacts()

# --- Routes ---

@app.route('/')
def index():
    """Renders the main prediction interface."""
    # Pass the mock categories to the HTML template for dropdown population
    return render_template('index.html', choices=CATEGORICAL_CHOICES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if not all([model, preprocessor, label_encoder]):
        return jsonify({'error': 'Model artifacts not loaded. Check server logs.'}), 500

    try:
        data = request.get_json(force=True)

        # 1. Extract features from request data
        input_data = {
            # Numerical features (converted to float)
            'handling_time': float(data['handling_time']),
            'response_to_survey_time': float(data['response_to_survey_time']),
            # Categorical features
            'channel_name': data['channel_name'],
            'category': data['category'],
            'subcategory': data['subcategory'],
            'agent_name': data['agent_name'],
            'supervisor': data['supervisor'],
            'manager': data['manager'],
            'tenurebucket': data['tenurebucket'],
            'agentshift': data['agentshift'],
            'handling_time_bucket': data['handling_time_bucket'],
        }

        # The order must match the expected order in your preprocessor's ColumnTransformer
        feature_names = [
            'handling_time', 'response_to_survey_time', 'channel_name',
            'category', 'subcategory', 'agent_name', 'supervisor',
            'manager', 'tenurebucket', 'agentshift', 'handling_time_bucket'
        ]

        # 2. Create DataFrame for transformation
        df_input = pd.DataFrame([input_data], columns=feature_names)

        # 3. Transform data using the loaded preprocessor
        X_transformed = preprocessor.transform(df_input)

        # 4. Predict the class index
        prediction_probs = model.predict(X_transformed)
        # Assuming classification model where prediction_probs is a probability distribution
        # Get the index of the highest probability
        predicted_index = prediction_probs.argmax(axis=1)[0]

        # 5. Decode the prediction using the Label Encoder
        predicted_csat = label_encoder.inverse_transform([predicted_index])[0]

        # 6. Return the result
        return jsonify({
            'prediction': predicted_csat,
            'status': 'success'
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # When running locally, categories need to be passed for templating,
    # but the environment for this file generation does not support Jinja,
    # so we rely on the frontend to define the CATEGORICAL_CHOICES for simplicity
    # and only use the backend for the prediction API.
    # The default index() route is primarily for context.
    # We will put the CATEGORICAL_CHOICES directly in the HTML for a single-file template solution.
    app.run(debug=True, host='0.0.0.0', port=5000)

