import mlflow
import dagshub
import logging
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.svm import SVR


df = pd.read_csv('../../data/processed/preprocessing_applied_dataset.csv')

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr_model = SVR(C=1, epsilon=0.01, gamma=0.1, kernel='rbf')

svr_model.train(X_train, X_test)

