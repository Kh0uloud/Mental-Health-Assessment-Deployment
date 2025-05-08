# 🧠 Mental Health Assessment: CI/CD Deployment Pipeline

This project automates the deployment of a mental health assessment model using a complete **MLOps pipeline**. It integrates **Flask**, **Docker**, **MLflow**, **Jenkins**, and **GitHub Actions** to streamline model deployment, version control, and operationalization of machine learning workflows.

The deployed model is a **Gradient Boosting Classifier** that predicts potential signs of depression based on user behavior and Twitter profile features.

> 📌 **Note**: Detailed data extraction, preprocessing, model development, and experimentation are handled in a separate repository:
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



