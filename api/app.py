from flask import Flask, request, jsonify, render_template
import mlflow
import pandas as pd
import os
import logging
from mlflow.pyfunc import load_model
from flask_basicauth import BasicAuth

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('AUTH_USERNAME', 'admin')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('AUTH_PASSWORD', 'password')

basic_auth = BasicAuth(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables and default values
MODEL_URI = os.getenv('MODEL_URI', 'models:/Mental_Health_assessment/1')
SERVER_PORT = os.getenv('PORT', '8000')
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'

# Load the model
try:
    model = load_model(MODEL_URI)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    """Endpoint to make Mental_Health_Assessment predictions."""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.form.to_dict()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    try:
        # Input validation
        required_fields = ['id', 'avg_retweets', 'avg_favorites', 'tweet_count', 'mean_time_gap',
                           'tweet_rate', 'unique_interactions', 'interactions_with_depressed',
                           'total_interactions', 'unique_sources', 'quote_ratio', 'followers_count',
                           'favourites_count', 'statuses_count', 'has_extended_profile', 
                           'profile_background_color_brightness', 'account_age']


        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields in input data'}), 400

        df = pd.DataFrame([data])
        df = df.drop('id', axis=1)
        prediction = model.predict(df)[0]  # Probability of class 1 (depression)
        logging.info(f"Prediction: {prediction}")
        is_depressed = prediction > 0.5

        # Log prediction and input data for monitoring
        logging.info(f"Input data: {data}")
        logging.info(f"Prediction: {prediction}, Is Depressed: {is_depressed}")

        return jsonify({
            'prediction': float(prediction),           # Convert NumPy float to native Python float
            'is_depressed': bool(is_depressed)         # Convert NumPy bool_ to native bool
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(SERVER_PORT), debug=DEBUG_MODE)