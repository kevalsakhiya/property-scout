import numpy as np
import pandas as pd
import re
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(file_path:str) -> pd.DataFrame:
    """
    Read the CSV data from the given file path.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info("Data read successfully from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error reading data: %s", e)
        raise

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataframe by extracting facilities and converting them to strings for vectorization.
    """
    try:
        # Dropping index 22 as it's bad data
        df.drop(22, inplace=True)

        # Converting all facilities in list
        df['TopFacilities'] = df['TopFacilities'].apply(extract_list)

        # Converting to string for vectorization
        df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)
        
        logger.info("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error("Error in preprocessing data: %s", e)
        raise

def extract_list(s:str) -> list:
    """
    Extract a list of facilities from a string.
    """
    return re.findall(r"'(.*?)'", s)

def vectorize_facilities(facilities_str:str) -> np.ndarray:
    """
    Vectorize the facilities string using TF-IDF.
    """
    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(facilities_str)
        logger.info("Facilities vectorization completed")
        return tfidf_matrix
    except Exception as e:
        logger.error("Error in vectorizing facilities: %s", e)
        raise

def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculate the cosine similarity matrix for the TF-IDF matrix.
    """
    try:
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info("Cosine similarity calculation completed")
        return cosine_sim
    except Exception as e:
        logger.error("Error in calculating cosine similarity: %s", e)
        raise

def recommend_properties(df:pd.DataFrame, property_name:str, cosine_sim:np.ndarray, top_n:int=5) -> pd.DataFrame:
    """
    Recommend properties based on the cosine similarity of facilities.
    """
    try:
        idx = df.index[df['PropertyName'] == property_name].tolist()[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        property_indices = [i[0] for i in sim_scores]
        
        recommendations_df = pd.DataFrame({
            'PropertyName': df['PropertyName'].iloc[property_indices],
            'SimilarityScore': [score[1] for score in sim_scores]
        })

        logger.info("Property recommendations generated for %s", property_name)
        return recommendations_df
    except Exception as e:
        logger.error("Error in recommending properties: %s", e)
        raise

def save_cosine_similarity(cosine_sim:np.ndarray, file_path:str) -> None:
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
        logger.info("Cosine similarity matrix saved to %s", file_path)
    except Exception as e:
        logger.error("Error in saving cosine similarity matrix: %s", e)
        raise

def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    file_path = f"{base_path}/data/raw/appartments.csv"
    cosine_sim_file_path = f"{base_path}/recommender_systems/ficility_cosine_similarity_matrix.joblib"
    
    df = read_data(file_path)
    df = preprocess_data(df)
    tfidf_matrix = vectorize_facilities(df['FacilitiesStr'])
    cosine_sim = calculate_cosine_similarity(tfidf_matrix)
    save_cosine_similarity(cosine_sim, cosine_sim_file_path)
    
    # # Example usage of recommend_properties function
    # property_name = 'Example Property'
    # recommendations = recommend_properties(df, property_name, cosine_sim)
    # print(recommendations)

if __name__ == "__main__":
    main()
