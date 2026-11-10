import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
ML_DIR = os.path.join(BASE_DIR, "ml")
MODELS_DIR = os.path.join(ML_DIR, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    mat_path = os.path.join(DATA_DIR, "student-mat.csv")
    por_path = os.path.join(DATA_DIR, "student-por.csv")

    df_mat = pd.read_csv(mat_path, sep=';')
    df_mat['course'] = 'Math'

    df_por = pd.read_csv(por_path, sep=';')
    df_por['course'] = 'Portuguese'

    df = pd.concat([df_mat, df_por], ignore_index=True)
    return df

def map_risk_level(g3):
    if g3 < 10:
        return 'High Risk'
    elif g3 <= 12:
        return 'Medium Risk'
    else:
        return 'Low Risk'

def map_binary_risk(g3):
    return 1 if g3 < 10 else 0

def prepare_data(df):
    df['risk_level'] = df['G3'].apply(map_risk_level)
    df['at_risk'] = df['G3'].apply(map_binary_risk)

    leakage_cols = ['G1', 'G2', 'G3', 'risk_level', 'at_risk']
    feature_cols = [c for c in df.columns if c not in leakage_cols]

    X = df[feature_cols].copy()
    y_binary = df['at_risk'].copy()
    y_multi = df['risk_level'].copy()

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    return X, y_binary, y_multi, categorical_cols, numeric_cols, feature_cols

def build_preprocessor(categorical_cols, numeric_cols):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    return preprocessor

def evaluate_model(clf, X_train, y_train, X_test, y_test, model_name="Model"):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            roc_auc = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "confusion_matrix": cm,
        "classification_report": report
    }

def main():
    print("--- 1. Loading and Preparing Data ---")
    df = load_data()
    print(f"Total student records: {len(df)}")
    print(f"Risk breakdown (Multi-class):\n{df['G3'].apply(map_risk_level).value_counts()}")
    print(f"Binary At-Risk (<10):\n{df['G3'].apply(map_binary_risk).value_counts()}")

    X, y_binary, y_multi, cat_cols, num_cols, feature_cols = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.20, random_state=42, stratify=y_binary
    )

    preprocessor = build_preprocessor(cat_cols, num_cols)

    models = {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            min_samples_split=6,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42
        )
    }

    results = {}
    fitted_pipelines = {}

    print("\n--- 2. Training and Evaluating Models (Holdout Test Set) ---")
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        eval_metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test, model_name=name)
        results[name] = eval_metrics
        fitted_pipelines[name] = pipe
        print(f"[{name}] Acc: {eval_metrics['accuracy']} | Recall: {eval_metrics['recall']} | Prec: {eval_metrics['precision']} | F1: {eval_metrics['f1_score']} | ROC-AUC: {eval_metrics['roc_auc']}")

    print("\n--- 3. Running 5-Fold Stratified Cross-Validation on Train Fold ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_summary = {}
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        cv_summary[name] = {
            "cv_accuracy_mean": round(float(np.mean(cv_res['test_accuracy'])), 4),
            "cv_recall_mean": round(float(np.mean(cv_res['test_recall'])), 4),
            "cv_precision_mean": round(float(np.mean(cv_res['test_precision'])), 4),
            "cv_f1_mean": round(float(np.mean(cv_res['test_f1'])), 4)
        }
        print(f"[{name} CV 5-Fold] Mean Recall: {cv_summary[name]['cv_recall_mean']} | Mean F1: {cv_summary[name]['cv_f1_mean']}")

    rf_pipe = fitted_pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps['classifier']

    cat_encoder = rf_pipe.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_names

    importances = rf_clf.feature_importances_
    feat_imp = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp[:15]]

    print("\n--- 4. Training Multi-Class Risk Model ---")
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=0.20, random_state=42, stratify=y_multi
    )
    multi_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42))
    ])
    multi_rf.fit(X_train_m, y_train_m)
    y_pred_m = multi_rf.predict(X_test_m)
    multi_acc = float(accuracy_score(y_test_m, y_pred_m))
    multi_f1 = float(f1_score(y_test_m, y_pred_m, average='weighted'))
    multi_report = classification_report(y_test_m, y_pred_m, output_dict=True, zero_division=0)
    print(f"[Multi-Class RF] Acc: {round(multi_acc, 4)} | Weighted F1: {round(multi_f1, 4)}")

    print("\n--- 5. Saving Artifacts ---")
    joblib.dump(fitted_pipelines["Random Forest"], os.path.join(MODELS_DIR, "rf_binary_pipeline.joblib"))
    joblib.dump(fitted_pipelines["Logistic Regression"], os.path.join(MODELS_DIR, "lr_binary_pipeline.joblib"))
    joblib.dump(multi_rf, os.path.join(MODELS_DIR, "rf_multiclass_pipeline.joblib"))

    metadata = {
        "dataset_stats": {
            "total_samples": len(df),
            "math_samples": len(df[df['course'] == 'Math']),
            "portuguese_samples": len(df[df['course'] == 'Portuguese']),
            "at_risk_count": int(df['at_risk'].sum()),
            "low_risk_count": int((1 - df['at_risk']).sum()),
            "at_risk_percentage": round(float(df['at_risk'].mean() * 100), 2),
            "multi_risk_counts": df['risk_level'].value_counts().to_dict()
        },
        "models_evaluation": results,
        "cv_summary": cv_summary,
        "top_feature_importances": top_features,
        "multiclass_evaluation": {
            "accuracy": round(multi_acc, 4),
            "f1_weighted": round(multi_f1, 4),
            "report": multi_report
        },
        "feature_columns": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "all": feature_cols
        }
    }

    with open(os.path.join(MODELS_DIR, "metrics_and_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Artifacts successfully saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
