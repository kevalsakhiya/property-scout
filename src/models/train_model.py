import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def split_data(df: pd.DataFrame, test_size:float=0.2,random_state:int=42):
    """
    Split the dataset into training and testing sets.
    """
    try:
        X = df.drop(columns=['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info("Data split into training and testing sets successfully.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> SVR:
    """
    Train the SVR model with pre-determined params
    """
    try:
        svr_model = SVR(C=1, epsilon=0.01, gamma=0.1, kernel='rbf')
        svr_model.fit(X_train, y_train)
        logger.info("Model trained successfully.")
        return svr_model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model: SVR, file_path: str) -> None:
    """
    Save the trained model to a file using joblib.
    """
    try:
        directory = Path(file_path).parent
        
        # Check if directory exists, if not, create it
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")

        joblib.dump(model, file_path)
        logger.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise

def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    data_file_path = f'{base_path}/data/processed/preprocessing_applied_dataset.csv'
    model_file_path = f"{base_path}/models/SVR_model.joblib"
    
    params = load_params(params_path='params.yaml')
    test_size = params['model-training']['test_size']
    random_state = params['model-training']['test_size']
    # Load data
    df = load_data(data_file_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df,test_size=float(test_size),random_state=int(random_state))
    
    # Train model
    svr_model = train_model(X_train, y_train)
    
    # Save model
    save_model(svr_model, model_file_path)

if __name__ == "__main__":
    main()