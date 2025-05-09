pipeline {
    agent any
    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/Kh0uloud/Mental-Health-Assessment-Deployment.git'
            }
        }
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t mentalhealth-assess .'
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh 'docker run mentalhealth-assess pytest tests/'
            }
        }
        stage('Push to Docker Registry') {
            steps {
                withDockerRegistry([credentialsId: 'docker-hub-credentials']) {
                    sh 'docker tag mentalhealth-assess kh0uloud/mentalhealth-assess:latest'
                    sh 'docker push kh0uloud/mentalhealth-assess:latest'
                }
            }
        }
        stage('Deploy Model') {
            steps {
                sh 'docker run -d -p 8000:8080 kh0uloud/mentalhealth-assess:latest'
            }
        }
    }
}