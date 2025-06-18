import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



mlflow.set_experiment("Telco Churn Prediction - Dagshub Autolog Run")

def load_processed_data(data_dir):
    """
    Memuat data yang sudah diproses (X_train, X_test, y_train, y_test) dari direktori yang diberikan.
    """
    try:
        if not os.path.exists(data_dir):
            print(f"Error: Direktori '{data_dir}' tidak ditemukan. Pastikan Anda telah menjalankan langkah preprocessing dan menyimpan data di lokasi ini.")
            return None, None, None, None

        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()

        print(f"Data yang diproses berhasil dimuat dari: {data_dir}")
        print(f"Dimensi data yang dimuat: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print(f"Error: Salah satu file (X_train.csv, X_test.csv, y_train.csv, y_test.csv) tidak ditemukan di '{data_dir}'.")
        return None, None, None, None
    except Exception as e:
        print(f"Error saat memuat data yang diproses: {e}")
        return None, None, None, None

def train_and_log_model_dagshub(X_train, y_train, X_test, y_test, params):
    """
    Melatih model dan mencatat eksperimen menggunakan MLflow autolog ke Dagshub.
    Menambahkan manual logging untuk memenuhi kriteria "2 nilai tambahan".
    """
    print("\nMelatih model dan melog ke Dagshub...")

    mlflow.sklearn.autolog() 
    
    mlflow.log_params(params)

    model = LogisticRegression(random_state=42, **params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.log_metric("num_features_in_model", X_train.shape[1])
    mlflow.log_metric("total_samples_trained", X_train.shape[0])
    print(f"Manually logged: num_features_in_model={X_train.shape[1]}, total_samples_trained={X_train.shape[0]}")

    mlflow.sklearn.log_model(model, "logistic_regression_model_autolog")

    print(f"Model berhasil dilatih.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    print("--- Memulai Pelatihan Model Machine Learning ke Dagshub ---")

    PATH_TO_PROCESSED_DATA = os.path.join(os.path.dirname(__file__), 'telco_churn_preprocessing')

    X_train, X_test, y_train, y_test = load_processed_data(PATH_TO_PROCESSED_DATA)

    if X_train is not None:
        model_params = {
            "C": 0.1,
            "solver": "liblinear",
            "max_iter": 100
        }

        train_and_log_model_dagshub(X_train, y_train, X_test, y_test, model_params)

        print("\nPeriksa Dagshub MLflow Tracking UI Anda untuk melihat hasil run ini.")

    else:
        print("\n--- Pelatihan Model Dibatalkan karena data tidak dapat dimuat. ---")
    print("\n--- Pelatihan Model ke Dagshub Selesai! ---")

