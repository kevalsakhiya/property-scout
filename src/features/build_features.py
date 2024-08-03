import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def categorize_luxury(score:int) -> None:
    """
    Categorize the luxury score into 'Low', 'Medium', or 'High'.
    """
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # For scores outside the defined bins

def categorize_floor(floor:int) -> None:
    """
    Categorize the floor number into 'Low Floor', 'Mid Floor', or 'High Floor'.
    """
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None  # For floors outside the defined bins

def encode_categorical_columns(df:pd.DataFrame) -> pd.DataFrame:
    """
    Apply Ordinal Encoding to categorical columns in the DataFrame.
    """
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        oe = OrdinalEncoder()
        df_encoded[col] = oe.fit_transform(df_encoded[[col]])

    return df_encoded

def build_features(data_file_path: Path) -> pd.DataFrame:
    """
    This Function perform feature engineering, and return a cleaned and transformed DataFrame.
    """
    try:
        # Load the data
        df = pd.read_csv(data_file_path)
        
        # Logging
        initial_shape = df.shape
        logger.info(f"Initial data shape: Rows = {initial_shape[0]}, Columns = {initial_shape[1]}")

        # Drop unnecessary columns
        train_df = df.drop(columns=['society', 'price_per_sqft'])
        logger.info(f"Dropped columns 'society' and 'price_per_sqft'. Shape after drop: {train_df.shape}")

        # Apply luxury score and floor number categorization
        train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)
        logger.info("Luxury score categorized successfully.")

        train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)
        logger.info("Floor number categorized successfully.")

        # Drop original 'floorNum' and 'luxury_score' columns
        train_df.drop(columns=['floorNum', 'luxury_score'], inplace=True)

        # Encode categorical columns
        data_label_encoded = encode_categorical_columns(train_df)

        # Separate features and target
        X_label = data_label_encoded.drop('price', axis=1)
        y_label = data_label_encoded['price']

        # Drop less important features
        export_df = X_label.drop(columns=['pooja room', 'study room', 'others'], errors='ignore')
        export_df['price'] = y_label
        final_shape = export_df.shape
        logger.info(f"Final data shape: Rows = {final_shape[0]}, Columns = {final_shape[1]}")

        # Log the difference in shape
        logger.info(f"Data reduced by: Rows = {initial_shape[0] - final_shape[0]}, Columns = {initial_shape[1] - final_shape[1]}")

        return export_df
    


    except Exception as e:
        logger.error(f"An error occurred during feature engineering: {e}")
        raise e

def save_df_to_csv(df: pd.DataFrame, file_path: str):
    """
    Save a pandas DataFrame to a CSV file. If the directory does not exist, create it.
    """
    try:
        # Convert file_path to a Path object
        file_path = Path(file_path)
        
        # Extract directory path from file path
        directory = file_path.parent
        
        # Check if directory exists, if not, create it
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")
        
        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        logger.info(f"DataFrame saved successfully to {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to save DataFrame to {file_path}: {e}")
        raise
 
if __name__ == "__main__":
    file_path = saving_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'cleaned' / 'gurgaon_properties_missing_value_imputation_v1.csv'
    feature_df = build_features(file_path)
    
    if feature_df is not None:
        saving_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'interim' / 'gurgaon_properties_post_feature_selection.csv'
        save_df_to_csv(feature_df,saving_path)
        logger.info(f"Data cleaned and saved to 'gurgaon_properties_missing_value_imputation_v1.csv'")