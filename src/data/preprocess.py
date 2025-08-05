import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline

def feature_engineering(X):
    X = X.copy()
    X['errorBalanceOrig'] = X['newbalanceOrig'] + X['amount'] - X['oldbalanceOrg']
    X['errorBalanceDest'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
    return X

def preprocess_data(data_path, base_model):
    data = pd.read_csv(data_path)
    data.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], inplace=True)
    
    X = data.drop('isFraud', axis=1)
    Y = data['isFraud']


    temp_X = feature_engineering(X)
    numerical_columns = [col for col in temp_X.columns if temp_X[col].dtype in ['int64', 'float64']]
    categorical_columns = [col for col in temp_X.columns if temp_X[col].dtype == 'O']

    fe_transform = FunctionTransformer(feature_engineering, validate=False)

    preprocess = ColumnTransformer(transformers=[
        ('scaling', StandardScaler(), numerical_columns),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_columns)
    ])

    pipeline = Pipeline(steps=[
        ('feature_engineer', fe_transform),
        ('preprocessing', preprocess),
        ('model', base_model)
    ])

    return pipeline, X, Y
