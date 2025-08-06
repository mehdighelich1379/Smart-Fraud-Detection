💳 Smart Fraud Detection Pipeline
Detecting fraudulent financial transactions using feature engineering, LightGBM, CatBoost, SMOTE, and evaluation visualizations.

📁 Project Structure
bash
Copy
Edit
fraud-detection/
├── data/                             ← (optional) zipped dataset or ignored raw data
├── images/                           ← Images used in reporting or Streamlit
├── mlruns/                           ← MLFlow tracking (excluded from Git)
├── notebook/
│   ├── EDA.ipynb                     ← Exploratory Data Analysis
│   ├── build_model.ipynb            ← Step-by-step experimentation
│   └── sampled_5m_with_fraud.csv    ← Sample dataset (if not ignored)
├── src/
│   ├── data/
│   │   └── preprocessing.py         ← Preprocessing + feature engineering
│   ├── models/
│   │   ├── train_model.py           ← Model training script
│   │   └── *.pkl                    ← Saved models (CatBoost / LGBM)
│   ├── utils/
│   │   └── metrics.py               ← Evaluation metrics and plots
│   └── init.py
├── app.py                           ← Streamlit dashboard entry point
├── explainder_dashboard.py          ← SHAP + feature importance visualizer
├── evaluate.py                      ← Model evaluation and reporting
├── main.py                          ← Full training pipeline
├── requirements.txt
└── README.md
🧠 Key Highlights
✅ End-to-End Pipeline:
Covers everything from EDA to deployment-ready models.

Modular and production-oriented design using src/ architecture.

📊 MLFlow Tracking:
All experiments logged under mlruns/

Keeps track of metrics, params, models, and artifacts.

📈 Model Types:
LightGBM with tuned parameters

CatBoost for additional benchmarking

⚙️ Feature Engineering:
Added domain-informed features to capture fraud behavior:

errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg

errorBalanceDest = oldbalanceDest + amount - newbalanceDest

📌 Oversampling with SMOTE:
Used SMOTE to balance the dataset.

Addressed class imbalance and improved recall.

🚦 Evaluation
Achieved near-perfect performance:

Metric	Score
Accuracy	~0.99
Precision	~0.99
Recall	~0.99
F1-Score	~0.99

Validated with 5-fold Stratified Cross-Validation

Visualized using confusion matrix and ROC AUC

📊 Visual Tools
explainder_dashboard.py for SHAP-based feature importance.

evaluate.py generates plots for precision-recall trade-offs, etc.

app.py (or streamlit_app.py) for optional interactive dashboard.

🚀 Getting Started
bash
Copy
Edit
# 1. Clone the repository
git clone https://github.com/mehdighelich1379/Smart-Fraud-Detection.git
cd Smart-Fraud-Detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py

# 4. Evaluate the model (optional)
python evaluate.py
🛠️ Tech Stack
Languages & Tools: Python, Jupyter, VS Code

ML Libraries: LightGBM, CatBoost, Scikit-learn, SMOTE

Visualization: Matplotlib, Seaborn, SHAP

Tracking: MLflow

📝 Final Thoughts
This pipeline demonstrates how a structured, iterative approach—especially domain-informed features—can drastically improve fraud detection performance even with imbalanced data.
This setup can be adapted for other anomaly detection tasks as well.


