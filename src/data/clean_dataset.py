# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_outliers(data_file_path: Path) -> pd.DataFrame:
    """
        Cleans the DataFrame by removing outliers and handling incorrect data.
    """

    try:
        # Creating dataframe from the file path
        df = pd.read_csv(data_file_path).drop_duplicates()

        initial_shape = df.shape
        logger.info(f"Initial data shape: Rows = {initial_shape[0]}, Columns = {initial_shape[1]}")

        # For price_per_sqft column taking only values lower or equal to 50000
        df = df[df['price_per_sqft'] <= 50000]

        # For area column taking values that are lower than 100000
        df = df[df['area'] < 100000]

        # Removing specific indexes due to data mismatch
        df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)

        # Manually assigning carpet area as area for specific rows
        corrections = {
            48: 115*9, 300: 7250, 2666: 5800, 1358: 2660,
            3195: 2850, 2131: 1812, 3088: 2160, 3444: 1175
        }
        for idx, area in corrections.items():
            if idx in df.index:
                df.loc[idx, 'area'] = area

        # For bedroom column removing all rows that have more than 10 bedrooms
        df = df[df['bedRoom'] <= 10]

        # Correcting specific row for carpet area
        if 2131 in df.index:
            df.loc[2131, 'carpet_area'] = 1812

        # Recalculating the price_per_sqft column after applying changes
        df['price_per_sqft'] = round((df['price'] * 10000000) / df['area'])

        final_shape = df.shape
        logger.info(f"Final data shape: Rows = {final_shape[0]}, Columns = {final_shape[1]}")
        logger.info(f"Data reduced by: {initial_shape[0] - final_shape[0]} rows and {initial_shape[1] - final_shape[1]} columns")

        return df

    except FileNotFoundError:
        logger.error(f"Error: The file {data_file_path} was not found.")
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file {data_file_path} is empty.")
    except pd.errors.ParserError:
        logger.error(f"Error: The file {data_file_path} could not be parsed.")
    except PermissionError:
        logger.error(f"Error: Permission denied while accessing {data_file_path}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")



def remove_missingdata(data_file_path: Path) -> pd.DataFrame:
    """
        Function to clean and impute missing data in a dataset
    """
    try:
        # Creating dataframe from the file path
        df = pd.read_csv(data_file_path).drop_duplicates()

        initial_shape = df.shape
        logger.info(f"Initial data shape: Rows = {initial_shape[0]}, Columns = {initial_shape[1]}")
    
        # checking where all three areas are present
        all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]

        # both super_built_up_area and carpet area present but built_up_area is null
        sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]
        adjusted_super_built_up_area = sbc_df['super_built_up_area'] / 1.105
        adjusted_carpet_area = sbc_df['carpet_area'] / 0.9

        # Compute the average of the adjusted values
        average_area = (adjusted_super_built_up_area + adjusted_carpet_area) / 2

        # Fill the missing values in 'built_up_area' with the computed average
        sbc_df.loc[sbc_df['built_up_area'].isna(), 'built_up_area'] = round(average_area[sbc_df['built_up_area'].isna()])
        
        # Updating dataframe
        df.update(sbc_df)

        # super_built_up_area present and other two is absent
        sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]
        fill_values = round(sb_df['super_built_up_area'] / 1.105)
        
        # Fill the missing values in 'built_up_area'
        sb_df.loc[sb_df['built_up_area'].isna(), 'built_up_area'] = fill_values[sb_df['built_up_area'].isna()]

        # Updating dataframe
        df.update(sb_df)

        # carpet_area is present and other two are null
        c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]

        fill_values = round(c_df['carpet_area']/0.9)

        # Fill the missing values in 'built_up_area'
        c_df.loc[c_df['built_up_area'].isna(), 'built_up_area'] = fill_values[c_df['built_up_area'].isna()]

        # Updating dataframe
        df.update(c_df)

        # making df of anomaly that we can detect in start of the plot
        anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]

        # Assigning area as builtup_are as it looks mistake
        anamoly_df['built_up_area'] = anamoly_df['area']

        # Updating dataframe
        df.update(anamoly_df)

        # Droping other area types now that we collected all data
        df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area'],inplace=True)

        median_value = int(df[df['property_type'] == 'house']['floorNum'].median())
        # Filling with median
        df.loc[df['floorNum'].isna(), 'floorNum'] = 2

        # Droping column as there is more than 28% values are missing
        df.drop(columns=['facing'],inplace=True)

        # Dropping row number 2536 as it's only one row
        df.drop(index=[2536],inplace=True)

        def mode_based_imputation(row):
            """
                Impute 'agePossession' based on the mode of similar rows with the same 'sector' and 'property_type'.
            """
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
        df['agePossession'] = df.apply(mode_based_imputation,axis=1)


        def mode_based_imputation2(row):
            """
            Impute 'agePossession' based on the mode of similar rows with the same 'sector'.
            """

            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
            
        df['agePossession'] = df.apply(mode_based_imputation2,axis=1)

        def mode_based_imputation3(row):
            """
                Impute 'agePossession' based on the mode of similar rows with the same 'property_type'.
            """
            if row['agePossession'] == 'Undefined':
                mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
                # If mode_value is empty (no mode found), return NaN, otherwise return the mode
                if not mode_value.empty:
                    return mode_value.iloc[0] 
                else:
                    return np.nan
            else:
                return row['agePossession']
            
        df['agePossession'] = df.apply(mode_based_imputation3,axis=1)

        # Log the final shape of the DataFrame
        final_shape = df.shape
        logger.info(f"Final data shape: Rows = {final_shape[0]}, Columns = {final_shape[1]}")

        # Log the difference in shape
        logger.info(f"Data reduced by: Rows = {initial_shape[0] - final_shape[0]}, Columns = {initial_shape[1] - final_shape[1]}")

        return df

    except FileNotFoundError:
        logger.error("The specified file was not found. Please check the file path and try again.")

    except pd.errors.EmptyDataError:
        logger.error("The specified file is empty. Please provide a valid data file.")

    except pd.errors.ParserError:
        logger.error("The specified file could not be parsed. Please check the file format and contents.")

    except KeyError as e:
        logger.error(f"Missing expected column in the data: {e}. Please check the data and try again.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


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

# Example usage
if __name__ == "__main__":
    
    logger.info(f"Removing Outliers from the data'")

    file_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw' / 'gurgaon_properties_cleaned_v2.csv'
    outlier_cleaned_df = remove_outliers(file_path)
    if outlier_cleaned_df is not None:
        saving_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'interim' / 'gurgaon_properties_outlier_treated_v1.csv'
        save_df_to_csv(outlier_cleaned_df,saving_path)
        
        logger.info(f"Data cleaned and saved to 'gurgaon_properties_cleaned_v3.csv'")
    
    logger.info(f"Removing missing data'")

    file_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'interim' / 'gurgaon_properties_outlier_treated_v1.csv'
    clean_missing_data = remove_missingdata(file_path)
    if clean_missing_data is not None:
        saving_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'interim' / 'gurgaon_properties_missing_value_imputation_v1.csv'
        save_df_to_csv(clean_missing_data,saving_path)
        logger.info(f"Data cleaned and saved to 'gurgaon_properties_missing_value_imputation_v1.csv'")
