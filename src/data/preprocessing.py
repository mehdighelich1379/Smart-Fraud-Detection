# preprocessing.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier

# ---------------------------
# 1. Feature Engineering
# ---------------------------
def feature_engineering(X):
    X = X.copy()

    # Feature engineering on balances
    X['errorBalanceOrig'] = X['newbalanceOrig'] + X['amount'] - X['oldbalanceOrg']
    X['errorBalanceDest'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
    X['diffOrig'] = X['oldbalanceOrg'] - X['newbalanceOrig']
    X['diffDest'] = X['newbalanceDest'] - X['oldbalanceDest']
    X['is_orig_empty_after'] = (X['newbalanceOrig'] == 0).astype(int)
    X['is_dest_empty_before'] = (X['oldbalanceDest'] == 0).astype(int)
    X['large_amount_flag'] = (X['amount'] > X['amount'].quantile(0.99)).astype(int)
    X['ratio_amount_balance'] = X['amount'] / (X['oldbalanceOrg'] + 1)

    # Drop 'step' if it exists
    if 'step' in X.columns:
        X = X.drop(columns=['step'])

    # Ensure time_period exists
    if 'time_period' not in X.columns:
        raise ValueError("Missing 'time_period' column. Please provide a 'time_period' feature (e.g., 'Morning', 'Afternoon', etc.)")


    return X


# ---------------------------
# 2. Build Preprocessing Pipeline
# ---------------------------
def build_pipeline(X, base_model=None):
    if base_model is None:
        base_model = CatBoostClassifier(verbose=0)

    # Step 1: Feature engineering transformer
    fe_transformer = FunctionTransformer(feature_engineering, validate=False)
    X_fe = feature_engineering(X)  # Preview transformed data to detect columns

    # Step 2: Column separation
    numerical_columns = [col for col in X_fe.columns if X_fe[col].dtype in ['int64', 'float64']]
    categorical_columns = [col for col in X_fe.columns if X_fe[col].dtype == 'O']

    # Step 3: ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('scaling', StandardScaler(), numerical_columns),
        ('ohe', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])

    # Step 4: Final pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineer', fe_transformer),
        ('preprocessing', preprocessor),
        ('model', base_model)
    ])

    return pipeline
