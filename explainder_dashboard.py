import pandas as pd
import joblib
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# 1. Load data
df = pd.read_csv('data/fraud_data/fraud.csv')

# 2. Sample balanced test data (1000 fraud, 1000 non-fraud)
fraud_df = df[df['isFraud'] == 1]
non_fraud_df = df[df['isFraud'] == 0]

fraud_sample = fraud_df.sample(n=1000, random_state=42)
non_fraud_sample = non_fraud_df.sample(n=1000, random_state=42)

dash_df = pd.concat([fraud_sample, non_fraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Load trained pipeline
pipeline = joblib.load('./src/models/fraud_catboost_pipeline.pkl')

# 4. Extract parts from pipeline
preprocessor = pipeline.named_steps['preprocessing']
feature_engineer = pipeline.named_steps['feature_engineer']
model = pipeline.named_steps['model']

# Add 'time_period' from 'step'
def convert_step_to_period(step):
    hour = step % 24
    if 0 <= hour < 6:
        return "midnight"
    elif 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    else:
        return "night"

# If 'step' exists, generate 'time_period'
if 'step' in dash_df.columns:
    dash_df['time_period'] = dash_df['step'].apply(convert_step_to_period)

# 5. Prepare data
X = dash_df.drop(columns=['isFraud'])
y = dash_df['isFraud']

# 6. Apply feature engineering
X_fe = feature_engineer.transform(X)

# 7. Apply preprocessing
X_processed_array = preprocessor.transform(X_fe)

# 8. Get feature names and create DataFrame
feature_names = preprocessor.get_feature_names_out()
X_processed = pd.DataFrame(X_processed_array, columns=feature_names)

# 9. Build explainer
explainer = ClassifierExplainer(model, X_processed, y)

# 10. Launch dashboard
ExplainerDashboard(explainer, title="Fraud Detection Explainer").run()

