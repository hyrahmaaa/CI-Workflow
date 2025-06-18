import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_processed_data(data_dir):
    """
    Memuat data yang sudah diproses (X_train, X_test, y_train, y_test) dari direktori yang diberikan.
    """
    print(f"DEBUG: Memulai load_processed_data. data_dir = '{data_dir}'") 

    try:
        if not os.path.exists(data_dir):
            print(f"ERROR: Direktori '{data_dir}' TIDAK ditemukan.") 
            return None, None, None, None

        print(f"DEBUG: Direktori '{data_dir}' DITEMUKAN. Mencoba membaca file...") 
        
        x_train_path = os.path.join(data_dir, 'X_train.csv')
        x_test_path = os.path.join(data_dir, 'X_test.csv')
        y_train_path = os.path.join(data_dir, 'y_train.csv')
        y_test_path = os.path.join(data_dir, 'y_test.csv')

        print(f"DEBUG: Cek X_train.csv: {os.path.exists(x_train_path)}")
        print(f"DEBUG: Cek X_test.csv: {os.path.exists(x_test_path)}")
        print(f"DEBUG: Cek y_train.csv: {os.path.exists(y_train_path)}")
        print(f"DEBUG: Cek y_test.csv: {os.path.exists(y_test_path)}")

        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        y_train = pd.read_csv(y_train_path).squeeze()
        y_test = pd.read_csv(y_test_path).squeeze()
        
        print(f"Data yang diproses berhasil dimuat dari: {data_dir}")
        print(f"Dimensi data yang dimuat: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as fnfe:
        print(f"ERROR: File tidak ditemukan di '{data_dir}'. Detail: {fnfe}") 
        return None, None, None, None
    except pd.errors.EmptyDataError as ede: 
        print(f"ERROR: Salah satu file CSV kosong atau tidak memiliki kolom. Detail: {ede}")
        return None, None, None, None
    except pd.errors.ParserError as pe: 
        print(f"ERROR: Terjadi kesalahan parsing saat membaca CSV. File mungkin rusak. Detail: {pe}")
        return None, None, None, None
    except Exception as e:
        print(f"ERROR: Terjadi kesalahan tak terduga saat memuat data yang diproses: {e}") 
        import traceback
        traceback.print_exc() 
        return None, None, None, None

def tune_and_log_model_manual(X_train, y_train, X_test, y_test, param_grid):
    print("\n--- Memulai Hyperparameter Tuning dan Logging Manual ---")



    model_base = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(
        estimator=model_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=1, 
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

    final_run_id = mlflow.active_run().info.run_id
    print(f"MLflow Run ID: {final_run_id}")
    
    with open("mlflow_run_id.txt", "w") as f:
        f.write(final_run_id)
    print(f"MLflow Run ID '{final_run_id}' berhasil disimpan ke mlflow_run_id.txt")

    print("--- Hyperparameter Tuning dan Logging Manual Selesai ---")


if __name__ == "__main__":
    print("##### Memulai Proses Hyperparameter Tuning Data Telco Churn #####")

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
