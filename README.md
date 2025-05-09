# 🧠 Mental Health Assessment: CI/CD Deployment Pipeline

This project delivers an end-to-end MLOps pipeline that operationalizes a machine learning model designed to assess the likelihood of depression based on social media activity.

The entire deployment process is containerized with Docker, tracked through MLflow, and orchestrated using CI/CD pipelines via Jenkins and GitHub Actions. This enables robust versioning, reproducibility, and automation; key components in deploying machine learning solutions at scale.

The deployed model is a Gradient Boosting Classifier, selected through comparative experimentation against other algorithms for its superior performance.

> 📌 **Note**: All data collection, preprocessing, feature engineering, and model experimentation are conducted in a separate repository: <br>
> 👉 [Mental Health Assessment - Modeling & Training](https://github.com/Kh0uloud/Modeling-Mental-Health-Trends-Using-Social-Media-Data)
---

### 📆 Installation & Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/Kh0uloud/Mental-Health-Assessment-Deployment.git
cd Mental-Health-Assessment-Deployment
```

#### 2. (Optional) Set Up a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate (On Linux/macOS: source venv/bin/activate)
```

#### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

### 🧪 Train and Evaluate the Model

Train, evaluate and save the model locally and log its metrics using:

```bash
python model/model.py
```
---

### 🚀 Run the Application Locally

#### 1. 📊 Start MLflow Tracking UI
To inspect experiment runs, metrics, and manage model versions:
```bash
mlflow ui
```
Once started, the MLflow UI will be available at: http://127.0.0.1:5000

Inside the UI:
* You can view the training runs and metrics
* Check if the model has been registered in the Model Registry
* If not registered, use the UI to register it manually

> ✅ This step is important because the Flask API will later load the model directly from the MLflow registry.

#### 2. 🌐 Run the Application Locally (Flask API)

```bash
python api/app.py
```

Visit the local app at: http://127.0.0.1:8000

---

### 🐳 Deploy with Docker

#### 1. Build the Docker Image

```bash
docker build -t mentalhealth-assess -f api/Dockerfile .
```

#### 2. Run the Docker Container

```bash
docker run -p 8000:8080 mentalhealth-assess
```

Visit the Dockerized app at: http://localhost:8080

---



