import argparse
import mlflow
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=5)
args = parser.parse_args()

lr = args.lr
epochs = args.epochs

mlflow.start_run()

# Log parameters
mlflow.log_param("learning_rate", lr)
mlflow.log_param("epochs", epochs)

accuracy = 0.0

for epoch in range(epochs):
    time.sleep(1)  # simulate training
    accuracy += random.uniform(0.05, 0.15)
    mlflow.log_metric("accuracy", accuracy, step=epoch)

mlflow.end_run()

print("Training completed")
print(f"Final accuracy: {accuracy}")
