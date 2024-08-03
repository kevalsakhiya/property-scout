import dagshub
import mlflow
import joblib
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
dagshub.init(repo_owner='kevalsakhiya', repo_name='property-scout', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kevalsakhiya/property-scout.mlflow')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load a machine learning model from a specified path.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load a dataset from a specified path.
    """
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}: {e}")
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a machine learning model using specified test data.
    """
    try:
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }
        logger.info("Model evaluation completed successfully")
        return metrics
    except Exception as e:
        logger.error(f"Failed to evaluate model: {e}")
        raise

def save_metrics(metrics: dict, metrics_path: str):
    """
    Save evaluation metrics to a JSON file.
    """
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved successfully to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics to {metrics_path}: {e}")
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        file_path = Path(file_path)
        # Extract directory path from file path
        directory = file_path.parent
        
        # Check if directory exists, if not, create it
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")

        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.info('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise


def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = f'{base_path}/models/SVR_model.joblib'
    data_path = f'{base_path}/data/processed/test.csv'
    metrics_path = f'{base_path}/reports/evaluation_metrics.json'
    
    mlflow.set_experiment('dvc-pipeline')

    with mlflow.start_run() as run:
        try:
            model = load_model(model_path)
            data = load_data(data_path)

            # log model params in mlflow
            if hasattr(model,'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # loging model in mlflow
            mlflow.sklearn.log_model(model,"model")

            save_model_info(run.info.run_id, model_path, 'reports/experiment_info.json')

            # making x_test and y_test
            X_test = data.iloc[:, :-1]
            y_test = data.iloc[:, -1]
            
            logger.info('Evaluating the model, This may take some time.')
            metrics = evaluate_model(model, X_test, y_test)

            # logging model metrics in mlflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Saving mlflow metrics in the report file
            save_metrics(metrics, metrics_path)

            # logging metric file to mlflow
            mlflow.log_artifact(metrics_path)
            
        except Exception as e:
            logger.error(f"An error occurred during the evaluation process: {e}")

if __name__ == "__main__":
    main()
