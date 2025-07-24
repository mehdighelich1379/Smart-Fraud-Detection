from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import joblib
from src.data.preprocess import preprocess_data

def training():
    data_path = r"C:\Users\T A T\project\Machin Learning\classification project\Fraud\data\fraud_data\fraud.csv"

    base_model = LGBMClassifier(
        class_weight='balanced',
        n_estimators=300,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=100,
        random_state=1
    )

    pipeline, X, y = preprocess_data(data_path, base_model)
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    pipeline.fit(x_train, y_train)
    joblib.dump(pipeline, r"C:\Users\T A T\project\Machin Learning\classification project\Fraud\src\models\trained_model.pkl")

    return pipeline, X, y, x_test, y_test
