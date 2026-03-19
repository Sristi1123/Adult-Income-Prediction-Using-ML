import mlflow
import mlflow.sklearn

def log_experiment(model, params, metrics, model_name):
    """Log experiments with MLflow"""
    mlflow.set_experiment("Income Prediction")
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        # Log metrics
        mlflow.log_metrics(metrics)
        # Log model
        mlflow.sklearn.log_model(model, model_name)