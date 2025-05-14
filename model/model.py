import os
os.environ['LOKY_MAX_CPU_COUNT'] = '16'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
import joblib
import logging
import sys
import mlflow
from mlflow.sklearn import log_model
import seaborn as sns
import matplotlib.pyplot as plt

import argparse

import tempfile

# Force MLflow to use a safe local folder inside the current workspace
local_artifact_dir = os.path.abspath("mlruns_artifacts")
os.makedirs(local_artifact_dir, exist_ok=True)


parser = argparse.ArgumentParser()
#parser.add_argument('--mlflow_tracking_uri', type=str, default='http://localhost:5000')  # Default MLflow URI
parser.add_argument('--model_name', type=str, default='Mental_Health_assessment')        # Default experiment name
args = parser.parse_args()

#mlflow.set_tracking_uri(args.mlflow_tracking_uri)
mlflow.set_experiment(args.model_name)


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

random_state = 42

def load_data(data_path):
    """
    Load data from a given path.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the data.
    """
    X = data.drop(["id", "depression"], axis=1)
    y = data["depression"]
    return X, y

def train_model(X, y, test_size=0.2, random_state=random_state):
    """
    Train the model and save the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.22, max_depth=5, subsample=0.7, min_samples_split=3, min_samples_leaf=4)
    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Compute metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    metrics['classification_report'] = classification_report(y_test, y_pred)
    
    # precision-recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    metrics['auprc'] = auprc

    return metrics

def save_model(model, model_save_path):
    """
    Save the model to a given path.
    """
    try:
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

if __name__ == '__main__':
    # mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(args.model_name)


    data_path = os.path.abspath(os.getenv('DATA_PATH', 'data/structured_data.csv'))
    model_save_path = os.path.abspath(os.getenv('MODEL_SAVE_PATH', 'model/saved_models/model.pkl'))

    # Load the dataset
    data = load_data(data_path)
    X, y = preprocess_data(data)

    with mlflow.start_run():
        mlflow.set_tag("mlflow.source.git.repoURL", "https://github.com/${{ github.repository }}")
        mlflow.set_tag("mlflow.source.name", __file__)

        # Train the model
        model, X_test, y_test = train_model(X, y)

        # Log model and parameters
        params = model.get_params()
        mlflow.log_params(params)

        # Evaluate model on simulated data
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        mlflow.log_metric("auprc", metrics['auprc'])

        # Log model
        log_model(model, "model")

        # Save the model locally
        save_model(model, model_save_path)

        logging.info(f"Model training completed with accuracy: {metrics['accuracy']}, ROC AUC: {metrics['roc_auc']}, AUPRC: {metrics['auprc']}")
        logging.info(metrics['classification_report'])