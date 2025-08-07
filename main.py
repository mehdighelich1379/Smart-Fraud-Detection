# main.py

import pandas as pd
from src.models.train_model import train_model
from src.utils.metrics import results

if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv("notebook/sampled_5m_with_fraud.csv") 


    # 2. Train model and get pipeline + splits
    pipeline, X, y, X_train, X_test, y_train, y_test = train_model(df)

    # 3. Evaluate the model
    results(
        pipeline=pipeline,
        x_train=X_train,
        x_test=X_test,
        y_test=y_test,
        threshold=0.5,
        X_full=X,
        Y_full=y
    )





    
