import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def main():
    columns_to_scale = ['property_type', 'bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
    columns_to_encode = ['sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
    price_column = ['price']

    data_path = Path(__file__).resolve().parent.parent.parent
    
    file_path = f"{data_path}/data/interim/gurgaon_properties_post_feature_selection.csv"

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
        
        logger.info("Saving the transformed dataframe")
        output_filepath = f"{data_path}/data/processed/preprocessing_applied_dataset.csv"

        logger.info("Saving dataframe to CSV file")
        df_transformed.to_csv(output_filepath, index=False)
        
        logger.info(f"Transformed data saved to {output_filepath}")
    
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()
