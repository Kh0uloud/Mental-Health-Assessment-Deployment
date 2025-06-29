pipeline {
    agent any
    environment {
        MLFLOW_TRACKING_URI = 'http://mlflow.example.com:8080'
        MODEL_NAME = 'mental-health-assessment'
        IMAGE_NAME = 'mentalhealth-assess'
        REGISTRY = 'host.docker.internal:6000'
    }
    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME .'
            }
        }
        stage('Tag Image') {
            steps {
                sh 'docker tag $IMAGE_NAME $REGISTRY/$IMAGE_NAME:latest'
            }
        }
        stage('Push to Local Docker Registry') {
            steps {
                sh 'docker push $REGISTRY/$IMAGE_NAME:latest'
            }
        }
        stage('Run Container') {
            steps {
                sh 'docker run -d --name mentalhealth-assess -p 5000:5000 $REGISTRY/$IMAGE_NAME:latest'
            }
        }
    }
}
