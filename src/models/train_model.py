import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from src.data.preprocessing import build_pipeline


def convert_step_to_period(step):
    hour = step % 24
    if 0 <= hour < 6:
        return "Midnight"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Night"



def train_model(df, test_size=0.2, random_state=42):
    """
    Train the model and return the trained pipeline and data splits.
    """

    # Filter valid transaction types
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT', 'PAYMENT', 'DEBIT', 'CASH_IN'])]

    # Create 'time_period' feature from 'step'
    df['time_period'] = df['step'].apply(convert_step_to_period)
    df = df.drop(columns=['step'])

    # Features and target
    X = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud' , 'isFraud'])
    y = df['isFraud']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Compute class weights
    classes = y_train.unique()
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

    # Define model
    base_model = CatBoostClassifier(
        max_depth=5,
        iterations=300,
        task_type='GPU',  # Remove this if you don't have GPU
        eval_metric='Recall',
        class_weights=class_weight_dict,
        random_state=random_state,
        verbose=100
    )

    # Build pipeline and train
    pipeline = build_pipeline(X_train, base_model=base_model)
    pipeline.fit(X_train, y_train)

    # Save trained pipeline
    joblib.dump(pipeline, "src/models/fraud_catboost_pipeline.pkl")

    return pipeline, X, y, X_train, X_test, y_train, y_test
