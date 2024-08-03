# -*- coding: utf-8 -*-
import logging
import zipfile
from pathlib import Path

def unzip_file(zip_file_path:Path, extract_dir_path:Path) -> None:
    """Extracts the data from the zip file and saves it into the data/raw folder."""
    # Logging setup
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('Starting to extract data set from zip file')
    
    try:
        # Checking if the extraction directory exists
        if not extract_dir_path.exists():
            logger.info(f"Creating directory: {extract_dir_path}")
            extract_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {extract_dir_path}")

        # Opening zip file and extracting all the data
        logger.info(f"Extracting {zip_file_path} to {extract_dir_path}")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir_path)
        
        logger.info(f'Files extracted successfully to {extract_dir_path}')
    except FileNotFoundError:
        logger.error(f"Error: The file {zip_file_path} was not found.")
    except zipfile.BadZipFile:
        logger.error(f"Error: The file {zip_file_path} is not a valid ZIP file.")
    except PermissionError:
        logger.error(f"Error: Permission denied while accessing {zip_file_path} or {extract_dir_path}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    zip_file_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'gurgaon_properties_cleaned_v2.zip'
    extract_dir_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'raw'
    unzip_file(zip_file_path, extract_dir_path)