pipeline {
    agent any

    stages {

        stage('Environment Check') {
            steps {
                bat 'python --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install --upgrade pip'
                bat 'python -m pip install scikit-learn joblib'
            }
        }

        stage('Train Model') {
            steps {
                bat 'python train.py'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'artifacts/**', fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Experiment 3: Jenkins automated ML training completed successfully'
        }
        failure {
            echo 'Experiment 3 failed'
        }
    }
}
