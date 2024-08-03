import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    )
logger = logging.getLogger(__name__)

def load_data(filepath:Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

def preprocess_data(df:pd.DataFrame, columns_to_scale:list, columns_to_encode:list,price_column:list) -> (np.ndarray,np.ndarray,ColumnTransformer):
    """Preprocess the data: scale numeric features and encode categorical features."""
    try:
        # Separating features and target variable
        X = df.drop(columns=price_column)
        y = df[price_column]
        
        # Defining the preprocessor for both numerical and categorical
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), columns_to_scale),
                ('cat', OneHotEncoder(drop='first'), columns_to_encode)
            ],
            remainder='passthrough'
        )

        preprocessor.fit(X)
        logger.info("Preprocessor fitted successfully.")

        X_transformed = preprocessor.transform(X).toarray()
        logger.info("Data transformed successfully.")

        y_transformed = np.log1p(y)
        
        return X_transformed, y_transformed, preprocessor
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

def create_transformed_df(X_transformed:np.ndarray, preprocessor:ColumnTransformer, columns_to_scale:list, columns_to_encode:list) -> pd.DataFrame:
    """Create a DataFrame with the transformed features."""
    try:
        num_feature_names = columns_to_scale
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(columns_to_encode)
        
        # Combining feature names for creating the dataframe
        all_feature_names = list(num_feature_names) + list(cat_feature_names)
        
        # Create a DataFrame with the transformed features
        df_transformed = pd.DataFrame(X_transformed, columns=all_feature_names)
        logger.info("Transformed DataFrame created successfully.")
        
        return df_transformed
    except Exception as e:
        logger.error(f"Error creating transformed DataFrame: {e}")
        raise

def save_data(df_transformed: pd.DataFrame,file_path:str, test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Splits the transformed DataFrame into train and test sets, then saves them as CSV files.
    """
    try:
        logger.info("Starting the data split process.")
        train_df, test_df = train_test_split(df_transformed, test_size=test_size, random_state=random_state)
        
        train_file_path = f"{file_path}/train.csv"
        test_file_path = f"{file_path}/test.csv"
        
        directory = Path(file_path).parent
        
        # Check if directory exists, if not, create it
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")

        # Saving data to csv
        logger.info(f"Saving train data to {train_file_path}.")
        train_df.to_csv(train_file_path, index=False)
        
        logger.info(f"Saving test data to {test_file_path}.")
        test_df.to_csv(test_file_path, index=False)
        
        logger.info("Data successfully saved.")
        
    except Exception as e:
        logger.error(f"An error occurred while saving the data: {e}")
        raise




def main():
    columns_to_scale = ['property_type', 'bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
    columns_to_encode = ['sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    price_column = ['price']

    base_path = Path(__file__).resolve().parent.parent.parent
    
    file_path = f"{base_path}/data/interim/gurgaon_properties_post_feature_selection.csv"

    try:
        logger.info("Loading data file")
        df = load_data(file_path)

        # Preprocess the data
        logger.info("Applying transformation")
        X_transformed, y_transformed, preprocessor = preprocess_data(df, columns_to_scale, columns_to_encode, price_column)

        # Create transformed DataFrame
        logger.info("Transforming sparse metrix back to dataframe")
        df_transformed = create_transformed_df(X_transformed, preprocessor, columns_to_scale, columns_to_encode)
        
        logger.info("Adding price column to the dataframe")
        df_transformed['price'] = y_transformed
        
        save_path = f"{base_path}/data/processed"
        save_data(df_transformed, save_path)

        logger.info("Saving the transformed dataframe")
        output_filepath = f"{base_path}/data/processed/preprocessing_applied_dataset.csv"

        logger.info("Saving dataframe to CSV file")
        df_transformed.to_csv(output_filepath, index=False)
        
        logger.info(f"Transformed data saved to {output_filepath}")
    
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
