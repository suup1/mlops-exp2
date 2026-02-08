pipeline {
    agent any

    stages {
        stage('Environment Check') {
            steps {
                bat 'java -version'
                bat 'python --version'
            }
        }

        stage('Train Model') {
            steps {
                bat 'python train.py'
            }
        }
    }
}
