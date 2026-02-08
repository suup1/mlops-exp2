import argparse
import mlflow
import time

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

mlflow.start_run()

mlflow.log_param("learning_rate", args.lr)
mlflow.log_param("epochs", args.epochs)

for epoch in range(args.epochs):
    loss = 1 / (epoch + 1)
    mlflow.log_metric("loss", loss)
    time.sleep(0.2)

mlflow.end_run()
