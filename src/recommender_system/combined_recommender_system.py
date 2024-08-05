import joblib
import numpy as np
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_joblib_file(file_path: str) -> np.ndarray:
    """
    Load a joblib file and return its content as a NumPy array.
    """
    try:
        # Attempt to load the joblib file
        logger.info(f"Loading joblib file from: {file_path}")
        data = joblib.load(file_path)
        
        logger.info("Joblib file loaded successfully")
        return data

    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise e

    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        raise e

    except Exception as e:
        logger.error(f"An error occurred while loading the joblib file: {str(e)}")
        raise e


def recommend_properties_with_scores(property_name:str, location_df:pd.DataFrame, attribute_similarity:np.ndarray, distance_similarity:np.ndarray, facility_similarity:np.ndarray, top_n:int=5):
    """
    Recommend properties based on the combined cosine similarity matrices.
    """
    try:
        cosine_sim_matrix = 30 * facility_similarity + 20 * distance_similarity + 8 * attribute_similarity
        idx = location_df.index.get_loc(property_name)
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]
        top_properties = location_df.index[top_indices].tolist()

        recommendations_df = pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })

        logger.info("Property recommendations generated for %s", property_name)
        return recommendations_df
    except Exception as e:
        logger.error("Error in recommending properties: %s", e)
        raise


def main():
    base_path = Path(__file__).resolve().parent.parent.parent
    attribute_similarity_path = f'{base_path}/recommender_systems/attribute_cosine_similarity_matrix.joblib'
    distance_similarity_path = f'{base_path}/recommender_systems/distance_cosine_similarity_matrix.joblib'
    facility_similarity_path = f'{base_path}/recommender_systems/ficility_cosine_similarity_matrix.joblib'
    location_df_path = f'{base_path}/recommender_systems/location_normalized_df.joblib'

    attribute_similarity = load_joblib_file(attribute_similarity_path)
    distance_similarity = load_joblib_file(distance_similarity_path)
    facility_similarity = load_joblib_file(facility_similarity_path)
    location_df = load_joblib_file(location_df_path)
#  test
    recommended_property = recommend_properties_with_scores('Ireo Victory Valley',location_df,attribute_similarity, distance_similarity, facility_similarity)
    print(recommended_property)


if __name__ == '__main__':
    main()