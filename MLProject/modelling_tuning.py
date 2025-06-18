import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

mlflow.set_experiment("Telco Churn Prediction - Tuned Model")

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

def tune_and_log_model_manual(X_train, y_train, X_test, y_test, param_grid):
    """
    Melakukan hyperparameter tuning menggunakan GridSearchCV dan mencatat hasilnya
    secara manual dengan MLflow.
    """
    # Pastikan baris ini sejajar dengan 'model_base = ...' atau pernyataan tingkat atas lainnya
    mlflow.sklearn.autolog(disable=True) 

    # Anda bisa tambahkan kembali baris print ini jika mau, sejajarkan indentasinya
    print("\n--- Memulai Hyperparameter Tuning dan Logging Manual ---") # Ini opsional, tapi bagus untuk log

    model_base = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(
        estimator=model_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    print("Melakukan Grid Search untuk hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print(f"Tuning selesai. Best Hyperparameters: {best_params}")
    print(f"Best CV Accuracy: {best_cv_score:.4f}")

    print("Logging parameter terbaik secara manual...")
    mlflow.log_params(best_params)

    mlflow.log_param("best_cv_accuracy", best_cv_score)

    y_pred_test = best_model.predict(X_test)
    y_prob_test = best_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, y_prob_test)

    print("Logging metrik test set secara manual...")
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    mlflow.log_metric("num_features", X_train.shape[1])
    mlflow.log_metric("num_train_samples", X_train.shape[0])
    print(f"Test Set Accuracy: {test_accuracy:.4f}")

    print("Logging model terbaik secara manual...")
    mlflow.sklearn.log_model(best_model, "tuned_logistic_regression_model")
    print("Model terbaik telah dilog ke MLflow.")

    # Ambil run_id dari run yang sudah aktif
    current_run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {current_run_id}")
    
    # Simpan run_id ke file
    with open("mlflow_run_id.txt", "w") as f:
        f.write(current_run_id)
    print(f"MLflow Run ID '{current_run_id}' berhasil disimpan ke mlflow_run_id.txt")

    print("--- Hyperparameter Tuning dan Logging Manual Selesai ---")


if __name__ == "__main__":
    print("##### Memulai Proses Hyperparameter Tuning Data Telco Churn #####")

    # PATH_TO_PROCESSED_DATA ini akan resolve ke MLProject/telco_churn_preprocessing
    PATH_TO_PROCESSED_DATA = os.path.join(os.path.dirname(__file__), 'telco_churn_preprocessing')

    X_train, X_test, y_train, y_test = load_processed_data(PATH_TO_PROCESSED_DATA)

    if X_train is not None:
        param_grid_lr = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [100, 200]
        }
        
        tune_and_log_model_manual(X_train, y_train, X_test, y_test, param_grid_lr)

        print("\nMLflow Tracking UI dapat dijalankan dengan perintah: mlflow ui")
        print(f"Akses di browser: http://localhost:5000")

    else:
        print("\n--- Proses Tuning Dibatalkan karena data tidak dapat dimuat. ---")
    print("\n##### Proses Hyperparameter Tuning Selesai! #####")
