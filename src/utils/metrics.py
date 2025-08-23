import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    make_scorer
)
from sklearn.model_selection import cross_validate, StratifiedKFold


def cross_val(model, X, Y):
    scoring = {
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary'),
        'accuracy': make_scorer(accuracy_score)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = cross_validate(
        model, X, Y,
        scoring=scoring,
        cv=cv,
        return_train_score=False
    )

    return pd.DataFrame(results)


def results(pipeline, x_train, x_test, y_test, threshold=0.5, X_full=None, Y_full=None):
    # 1. Prediction & Thresholding
    y_pred_probs = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_probs > threshold).astype(int)

    # 2. Classification Report
    print("🔍 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('🧩 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('📈 ROC Curve')
    plt.legend()
    plt.show()

    # 5. Feature Importances
    try:
        x_fe = pipeline.named_steps['feature_engineer'].transform(x_train)
        x_processed = pipeline.named_steps['preprocessing'].transform(x_fe)
        raw_feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
        clean_feature_names = [name.replace("scaling__", "").replace("ohe__", "") for name in raw_feature_names]

        importances = pipeline.named_steps['model'].feature_importances_
        feature_series = pd.Series(importances, index=clean_feature_names).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_series.values, y=feature_series.index)
        plt.title("🔍 Feature Importances (from x_train)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"⚠️ Could not plot feature importances: {e}")

    # 6. Cross-Validation (optional)
    if X_full is not None and Y_full is not None:
        print("🔁 Cross-validation on full dataset (StratifiedKFold, 5 folds):")
        try:
            df_cv = cross_val(pipeline, X_full, Y_full)
            display = df_cv.mean()[['test_precision', 'test_recall', 'test_f1', 'test_accuracy']]
            print(display.round(4))
        except Exception as e:
            print(f"⚠️ Cross-validation failed: {e}")
    else:
        print("ℹ️ Full dataset not provided. Skipping cross-validation.")




