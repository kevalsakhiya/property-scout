property-scout
==============================

## Overview
This project aims to develop a sophisticated real-estate machine learning model designed to analyze extensive property data, including apartments, homes, and villas, and provide insightful recommendations. The model assists individuals in making informed real-estate decisions by suggesting the best properties and locations that meet their needs.

## Project Structure

### Goals
- **Provide suggestions based on user's requirements**: The system recommends properties based on various user-defined criteria.
- **Model Used**: Support Vector Regression (SVR) with hyperparameter tuning.
- **Data Tracking**: Implemented via a DVC pipeline, which can be executed using the command `dvc repro`.
- **Experiment Tracking**: Managed with MLflow.
- **Collaboration and Version Control**: Connected with DagsHub for seamless integration and tracking.

## Recommendation System
The recommendation system operates using three distinct methods to ensure comprehensive suggestions:

1. **Attribute-Based Recommendation**: 
    - Calculates similarity scores based on property attributes such as price, size, type of apartment, etc.
    - Uses cosine similarity to measure the likeness between properties based on these attributes.
  
2. **Facility-Based Recommendation**: 
    - Determines similarity based on nearby facilities and amenities, such as schools, hospitals, parks, and shopping centers.
    - Uses cosine similarity to evaluate how similar properties are in terms of the availability and proximity of these facilities.
  
3. **Distance-Based Recommendation**: 
    - Computes similarity based on the distance from a given location, which could be the user's workplace, a city center, or any other point of interest.
    - Uses cosine similarity to assess the spatial relationship between properties and the specified location.

The final recommendation combines results from all three methods, leveraging cosine similarity, to provide the most accurate and relevant suggestions. This multi-faceted approach ensures that the recommendations are not only based on property characteristics but also consider the surrounding environment and convenience for the user.

### Features
- **User-Centric Recommendations**: Tailored suggestions based on individual user preferences.
- **Comprehensive Analysis**: Utilizes multiple criteria to ensure well-rounded recommendations.
- **Efficient Data and Experiment Tracking**: Leveraging DVC and MLflow for robust tracking and reproducibility.

### How it Works
1. **Data Collection and Preprocessing**: The property data is collected, cleaned, and preprocessed to ensure it is ready for analysis. This includes handling missing values, normalizing numerical features, and encoding categorical variables.
  
2. **Model Training**: The Support Vector Regression (SVR) model is trained using the processed data. Hyperparameter tuning is performed to optimize the model's performance.
  
3. **Similarity Calculation**: For recommendation purposes, the system calculates similarity scores between properties using cosine similarity. This involves computing the cosine of the angle between property feature vectors in a multi-dimensional space, providing a measure of how similar the properties are to each other.

4. **Combining Recommendations**: The system integrates the results from attribute-based, facility-based, and distance-based recommendations to generate a comprehensive list of suggested properties for the user.

### Running the Pipeline
To run the data processing and model training pipeline, use:
```sh
dvc repro
```


# This is the folder structure of the project.
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    |── recommender_systems<- Cosine numpy matrixs to calculate similarity
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    |   |── recommender_systems <- Scripts to make recommender system.
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
