# Fraud Detection Pipeline  
Detecting fraudulent financial transactions using advanced feature engineering and LightGBM.  

## 📁 Project Structure

```bash
fraud-detection/
├── notebooks/
│   └── build_model.ipynb       ← Step-by-step experimentation
├── src/
│   ├── data/
│   │   └── preprocess.py       ← Feature engineering + preprocessing pipeline
│   ├── models/
│   │   └── train_model.py      ← Model training script with LightGBM
│   ├── utils/
│   │   └── metrics.py          ← Evaluation: metrics, confusion matrix, ROC
│   └── init.py
├── app/
│   └── streamlit_app.py        ← (Optional) Streamlit dashboard
├── evaluate.py                 ← Script for model performance visualization
├── main.py                     ← Run training + evaluation together
├── requirements.txt
└── README.md

---

## 🔍 Project Description

This project builds an end-to-end pipeline for credit card fraud detection. It applies robust preprocessing, feature engineering, and a tuned LightGBM model to detect fraud with high recall and balanced precision. The process starts with Exploratory Data Analysis (EDA), followed by various modeling stages to find the most stable and generalizable solution.

---

## ✅ Steps Performed

### 1. EDA (notebooks/EDA.ipynb)  
- Analyzed distribution of fraud vs non-fraud samples (highly imbalanced).
- Explored transaction types, amount distributions, and balance inconsistencies.
- Visualized correlations and outliers.

### 2. Initial Model (No class weights)
- Trained LightGBM on raw data.
- Result: Low recall on fraud class (missed most frauds).

### 3. Model with Class Weights  
- Applied class_weight='balanced' to improve fraud detection.
- Result: Recall improved, but precision dropped drastically (too many false positives).

### 4. **Using scale_pos_weight / is_unbalance in LGBM**
- Result: Still unstable performance.

###Smote Oversamplingng**
- Used to synthetically balance classes.
- Slight improvement, precision still too lowow**.

###🧠 Feature Engineering (Key Step!)!)**
- Introduced 2 engineered features:
  - errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
  - errorBalanceDest = oldbalanceDest + amount - newbalanceDest
- These features capture inconsistencies that typically occur in fraudulent transactions.
- After retraining the model, we achieved:
Precision:n:** ~0.99  
Recall:l:** ~0.99  
F1-score:e:** ~0.99  
Accuracy:y:** ~0.99  
- Balanced results with strong generalization validated 5-fold Stratified Cross-Validationon**.

---

## 🛠️ Technologies Used
- Python, Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn, LightGBM
- Joblib, Jupyter Notebook

---

## 🚀 How to Run

1. Clone the repo:  
   `bash
   git clone https://github.com/mehdighelich1379/Smart-Fraud-Detection.git
   cd fraud-detection-model

2. Install dependencies:

pip install -r requirements.txt


3. Train the model:

python main.py


4. Evaluate the model (optional):

python evaluate.py




---

📈 Final Notes

This project demonstrates how careful feature engineering and iterative evaluation (precision vs recall trade-offs) can significantly improve fraud detection systems. While data imbalance is a major challenge, domain-driven feature design often yields the best boost in model performance.
