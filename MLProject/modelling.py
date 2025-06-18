import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns # Pastikan seaborn diinstal via conda.yaml

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer

# --- KONFIGURASI DAGSHUB/MLFLOW TRACKING ---
# Ambil dari environment variables yang disetel di GitHub Actions,
# atau gunakan nilai default jika dijalankan lokal tanpa env vars.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "https://dagshub.com/hyrahmaaa/Submission-Membangun-Sistem-Machine-Learning.mlflow")
MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME", "hyrahmaaa")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD", "568d3a44cb143c40099b002d1e13b8429305e1d6") # Pastikan ini token Dagshubmu

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# Set environment variables untuk autentikasi Dagshub (penting untuk push ke Dagshub)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

# Set nama eksperimen MLflow
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

def train_and_log_model_dagshub(X_train, y_train, X_test, y_test):
    """
    Melatih model (dengan tuning) dan mencatat eksperimen menggunakan MLflow ke Dagshub,
    termasuk logging manual dan artifact plotting.
    """
    print("\nMelatih model (dengan tuning) dan melog ke Dagshub...")

    mlflow.sklearn.autolog() # Aktifkan autolog untuk menangkap param, metrik, model dasar

    with mlflow.start_run() as run: # Mulai MLflow Run secara eksplisit
        current_run_id = run.info.run_id
        print(f"MLflow Run ID: {current_run_id}")

        # --- Hyperparameter Tuning dengan GridSearchCV ---
        model_base = LogisticRegression(random_state=42)
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [100, 200]
        }
        # Menggunakan make_scorer dengan f1_score untuk klasifikasi
        scoring_metric = make_scorer(f1_score) 

        grid_search = GridSearchCV(
            estimator=model_base,
            param_grid=param_grid,
            cv=3, # 3-fold cross-validation
            scoring=scoring_metric, # Metrik scoring
            verbose=1,
            n_jobs=-1 # Gunakan semua core yang tersedia
        )

        print("\nMemulai hyperparameter tuning dengan GridSearchCV...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        # Mengambil best score dari GridSearchCV
        best_cv_score = grid_search.best_score_ 

        print(f"\nParameter terbaik ditemukan: {best_params}")
        print(f"F1-Score terbaik dari Cross-Validation: {best_cv_score:.4f}")

        # Log parameter terbaik dan detail tuning ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LogisticRegression_Tuned")
        mlflow.log_param("cv_folds", grid_search.cv)
        mlflow.log_param("grid_search_scoring", "f1_score") # Log nama metrik scoring yang dipakai

        # Log metrik dari cross-validation
        mlflow.log_metric("best_cv_f1_score", best_cv_score)
        
        # --- Evaluasi Model Terbaik di Test Set ---
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\n--- Hasil Evaluasi Model Terbaik di Test Set ---")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        # Log metrik ke MLflow (manual logging)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # Log 2 nilai tambahan (dari kebutuhan tugas)
        mlflow.log_metric("num_features_in_model", X_train.shape[1])
        mlflow.log_metric("total_samples_trained", X_train.shape[0])
        print(f"Manually logged: num_features_in_model={X_train.shape[1]}, total_samples_trained={X_train.shape[0]}")

        # Log Model ke MLflow
        # 'tuned_logistic_regression_model' adalah nama model di MLflow artifacts
        mlflow.sklearn.log_model(best_model, "tuned_logistic_regression_model")
        
        # --- Plotting Koefisien Fitur (Mirip Feature Importance untuk LR) ---
        # Ini akan membuat plot dan menyimpannya sebagai artifact
        if hasattr(best_model, 'coef_') and len(best_model.coef_.shape) == 2:
            coefficients = pd.Series(best_model.coef_[0], index=X_train.columns)
            coefficients_abs = coefficients.abs().sort_values(ascending=True) # Sort berdasarkan nilai absolut
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(x=coefficients_abs.nlargest(20).values, y=coefficients_abs.nlargest(20).index, ax=ax, palette="viridis")
            ax.set_title("Top 20 Absolute Feature Coefficients (Tuned Logistic Regression)")
            ax.set_xlabel("Absolute Coefficient Value")
            ax.set_ylabel("Feature")
            fig.tight_layout()
            
            plot_path = "logistic_regression_coefficients.png"
            fig.savefig(plot_path)
            plt.close(fig) # Penting untuk menutup plot
            mlflow.log_artifact(plot_path, "feature_plots")
            os.remove(plot_path) # Hapus file lokal setelah di-log

        print("--- Logging ke MLflow Selesai ---")
        print(f"Link ke Dagshub Run: {mlflow.active_run().info.artifact_uri.split('/artifacts')[0]}")

        # --- Tulis Run ID ke file agar GitHub Actions bisa membacanya ---
        # File ini akan ada di root directory runner, yaitu /home/runner/work/CI-Workflow/CI-Workflow/
        run_id_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlflow_run_id.txt')
        with open(run_id_file_path, "w") as f:
            f.write(current_run_id)
        print(f"Run ID {current_run_id} disimpan ke {run_id_file_path}")


if __name__ == "__main__":
    print("--- Memulai Pelatihan Model Machine Learning ke Dagshub ---")

    # Pastikan 'telco_churn_preprocessing' adalah folder di dalam MLProject
    PATH_TO_PROCESSED_DATA = os.path.join(os.path.dirname(__file__), 'telco_churn_preprocessing')

    X_train, X_test, y_train, y_test = load_processed_data(PATH_TO_PROCESSED_DATA)

    # Pastikan data berhasil dimuat sebelum melanjutkan pelatihan
    if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
        train_and_log_model_dagshub(X_train, y_train, X_test, y_test)

        print("\nPeriksa Dagshub MLflow Tracking UI Anda untuk melihat hasil run ini.")
        print("Anda juga akan menemukan file 'mlflow_run_id.txt' di artifact GitHub Actions.")

    else:
        print("\n--- Pelatihan Model Dibatalkan karena data tidak dapat dimuat. ---")
    print("\n--- Pelatihan Model ke Dagshub Selesai! ---")
