import numpy as np
import pandas as pd
import re
import json
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(file_path:str) -> pd.DataFrame:
    """
    Read the CSV data from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Data read successfully from %s", file_path)
        df.drop(22, inplace=True)
        return df
    except Exception as e:
        logger.error("Error reading data: %s", e)
        raise


def refined_parse_modified_v2(detail_str:str) -> dict:
    """
    Parse the price details string and extract information into a dictionary.
    """
    try:
        details = json.loads(detail_str.replace("'", "\""))
    except json.JSONDecodeError as e:
        logging.error("Error decoding JSON: %s", e)
        return {}

    extracted = {}
    for bhk, detail in details.items():
        # Extract building type
        extracted[f'building_type_{bhk}'] = detail.get('building_type')

        # Parsing area details
        area = detail.get('area', '')
        area_parts = area.split('-')
        if len(area_parts) == 1:
            try:
                value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area_low_{bhk}'] = value
                extracted[f'area_high_{bhk}'] = value
            except ValueError:
                extracted[f'area_low_{bhk}'] = None
                extracted[f'area_high_{bhk}'] = None
        elif len(area_parts) == 2:
            try:
                extracted[f'area_low_{bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                extracted[f'area_high_{bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
            except ValueError:
                extracted[f'area_low_{bhk}'] = None
                extracted[f'area_high_{bhk}'] = None

        # Parsing price details
        price_range = detail.get('price-range', '')
        price_parts = price_range.split('-')
        if len(price_parts) == 2:
            try:
                extracted[f'price_low_{bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                extracted[f'price_high_{bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                if 'L' in price_parts[0]:
                    extracted[f'price_low_{bhk}'] /= 100
                if 'L' in price_parts[1]:
                    extracted[f'price_high_{bhk}'] /= 100
            except ValueError:
                extracted[f'price_low_{bhk}'] = None
                extracted[f'price_high_{bhk}'] = None

    return extracted

def preprocess_property_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess property data by parsing price details and constructing a refined dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the property data.

    Returns:
    pd.DataFrame: The refined dataframe.
    """
    data_refined = []

    for _, row in df.iterrows():
        features = refined_parse_modified_v2(row['PriceDetails'])
        
        # Construct a new row for the transformed dataframe
        new_row = {'PropertyName': row['PropertyName']}
        
        # Populate the new row with extracted features
        for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
            new_row[f'building_type_{config}'] = features.get(f'building_type_{config}')
            new_row[f'area_low_{config}'] = features.get(f'area_low_{config}')
            new_row[f'area_high_{config}'] = features.get(f'area_high_{config}')
            new_row[f'price_low_{config}'] = features.get(f'price_low_{config}')
            new_row[f'price_high_{config}'] = features.get(f'price_high_{config}')
        
        data_refined.append(new_row)

    df_refined = pd.DataFrame(data_refined).set_index('PropertyName')
    df_refined['building_type_Land'] = df_refined['building_type_Land'].replace({'': 'Land'})

    return df_refined

def encode_and_normalize(df:pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical columns and normalize the dataframe.
    """
    try:
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        df_ohe = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        df_ohe.fillna(0, inplace=True)

        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_ohe), columns=df_ohe.columns, index=df_ohe.index)
        
        logging.info("Data encoding and normalization completed")
        return df_normalized
    except Exception as e:
        logging.error("Error in encoding and normalizing data: %s", e)
        raise

def calculate_cosine_similarity(df:pd.DataFrame) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for the given dataframe.
    """
    try:
        cosine_sim = cosine_similarity(df)
        logging.info("Cosine similarity calculation completed")
        return cosine_sim
    except Exception as e:
        logging.error("Error in calculating cosine similarity: %s", e)
        raise

def recommend_properties_with_scores(df:pd.DataFrame, property_name:str, cosine_sim:np.ndarray, top_n:int=5) -> pd.DataFrame:
    """
    Recommend properties based on the cosine similarity of type and price.
    """
    try:
        idx = df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = df.index[top_indices].tolist()

        recommendations_df = pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })

        logging.info("Property recommendations generated for %s", property_name)
        return recommendations_df
    except Exception as e:
        logging.error("Error in recommending properties: %s", e)
        raise

def save_cosine_similarity(cosine_sim:np.ndarray, file_path:str):
    """
    Save the cosine similarity matrix to a file using joblib.
    """
    try:
        directory = Path(file_path).parent
         # Check if directory exists, if not, create it
        if not directory.exists():
            logger.info(f"Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {directory}")

        joblib.dump(cosine_sim, file_path)
        logging.info("Cosine similarity matrix saved to %s", file_path)
    except Exception as e:
        logging.error("Error in saving cosine similarity matrix: %s", e)
        raise

def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    file_path = f"{base_path}/data/raw/appartments.csv"
    cosine_sim_file_path = f"{base_path}/recommender_systems/attribute_cosine_similarity_matrix.joblib"

    
    df = read_data(file_path)
    df_refined = preprocess_property_data(df)
    df_normalized = encode_and_normalize(df_refined)
    cosine_sim = calculate_cosine_similarity(df_normalized)
    save_cosine_similarity(cosine_sim, cosine_sim_file_path)
if __name__ == "__main__":
    main()
