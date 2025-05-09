import mlflow 
import pandas as pd 
from flask import Flask, request, jsonify
import argparse
from mlflow.tracking import MlflowClient


def get_latest_model_uri(model_name: str) -> str:
    """Get the latest registered model version"""
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if not model_versions:
        raise ValueError(f"No versions found for model: {model_name}")
    latest_model_version = max(model_versions, key=lambda x: x.version)
    return f"models:/{model_name}/{latest_model_version.version}"

def create_app(mlflow_tracking_uri: str, model_name: str):
    """Initialize the Flask app and load the latest Mlflow model."""
    app = Flask(__name__)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    model_uri = get_latest_model_uri(model_name)
    model = mlflow.pyfunc.load_model(model_uri)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            """Predict endpoint for the Flask app."""
            data = request.get_json(force=True)
            required_fields = ['id', 'avg_retweets', 'avg_favorites', 'tweet_count', 'mean_time_gap',
                            'tweet_rate', 'unique_interactions', 'interactions_with_depressed',
                            'total_interactions', 'unique_sources', 'quote_ratio', 'followers_count',
                            'favourites_count', 'statuses_count', 'has_extended_profile', 
                            'profile_background_color_brightness', 'account_age']

            input_df = pd.DataFrame([{key: float(data[key]) for key in required_fields}])
            predictions = model.predict(input_df).tolist()
            return jsonify({"predictions":predictions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app for serving MLflow model")
    parser.add_argument('--mlflow-tracking-uri', type=str, default='http://mlflow.example.com:8080', help="MLflow tracking URI")
    parser.add_argument('--model-name', type=str, default='Mental_Health_assessment', help="Name of the registered model")
    args = parser.parse_args()

    app = create_app(args.mlflow_tracking_uri, args.model_name)
    app.run(host='0.0.0.0', port=5000)