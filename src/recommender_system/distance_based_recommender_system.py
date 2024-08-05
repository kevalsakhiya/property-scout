import ast
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re

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
def distance_to_meters(distance_str:str) -> float:
    """
    Convert distance string to meters.
    """
    try:
        match = re.search(r"(\d+(\.\d+)?)\s*(Km|KM|km|Kms|Meter|meter|mtrs|m)", distance_str)
        if match:
            value = float(match.group(1))
            unit = match.group(3)
            if (unit.lower() == 'km') or (unit.lower() == 'kms'):
                return value * 1000
            elif (unit.lower() == 'meter') or (unit.lower() == 'm') or (unit.lower() == 'mtrs'):
                return value
        return None
    except Exception as e:
        logger.error("Error converting distance to meters: %s", e)
        return None

def extract_location_distances(df:pd.DataFrame) -> pd.DataFrame:
    """
    Extract and convert distances for each location from the dataframe.
    """
    location_matrix = {}
    for index, row in df.iterrows():
        distances = {}
        for location, distance in ast.literal_eval(row['LocationAdvantages']).items():
            distances[location] = distance_to_meters(distance)
        location_matrix[index] = distances

    location_df = pd.DataFrame.from_dict(location_matrix, orient='index')
    location_df.index = df.PropertyName
    location_df.fillna(54000, inplace=True)  # Replacing NaN with a large value

    return location_df

def normalize_dataframe(df:pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the dataframe using StandardScaler to put all values on same scale.
    """
    try:
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        logger.info("Data normalization completed")
        return df_normalized
    except Exception as e:
        logger.error("Error in normalizing data: %s", e)
        raise

def calculate_cosine_similarity(df:pd.DataFrame) -> np.ndarray:
    """
    Calculate the cosine similarity matrix for the given dataframe.
    """
    try:
        cosine_sim = cosine_similarity(df)
        logger.info("Cosine similarity calculation completed")
        return cosine_sim
    except Exception as e:
        logger.error("Error in calculating cosine similarity: %s", e)
        raise

def save_array(np_array:np.ndarray, file_path:str) -> None:
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
        joblib.dump(np_array, file_path)
        logger.info("Cosine similarity matrix saved to %s", file_path)
    except Exception as e:
        logger.error("Error in saving cosine similarity matrix: %s", e)
        raise

def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    file_path = f"{base_path}/data/raw/appartments.csv"
    cosine_sim_file_path = f"{base_path}/recommender_systems/distance_cosine_similarity_matrix.joblib"
    location_df_file_path = f"{base_path}/recommender_systems/location_normalized_df.joblib"

    
    df = read_data(file_path)
    location_df = extract_location_distances(df)
    location_df_normalized = normalize_dataframe(location_df)
    cosine_sim = calculate_cosine_similarity(location_df_normalized)

    save_array(cosine_sim, cosine_sim_file_path)
    save_array(location_df_normalized, location_df_file_path)
    
if __name__ == "__main__":
    main()
