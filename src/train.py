import argparse
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    # -------- Argument parsing --------
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs

    # -------- Load data --------
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Model --------
    model = LogisticRegression(
        C=1.0 / lr,
        max_iter=epochs,
        solver="lbfgs",
        multi_class="auto"
    )

    # -------- Train --------
    model.fit(X_train, y_train)

    # -------- Evaluate --------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # -------- Log to MLflow --------
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("epochs", epochs)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")

    print(f"Training completed | accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
