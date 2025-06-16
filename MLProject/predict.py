# MLProject/predict.py
import requests
import pandas as pd
import json
import os 

# === Fungsi untuk melakukan prediksi (ini tetap di luar __main__) ===
def make_prediction_request(data_sample, url, headers):
    payload = {
        "dataframe_split": data_sample.to_dict(orient="split")
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()

if __name__ == "__main__":
    # === Kode di bawah ini hanya akan dieksekusi ketika predict.py dijalankan langsung ===

    # === Load preprocessed CSV untuk inference ===
    script_dir = os.path.abspath(os.path.dirname(__file__))
    X_test_data_path = os.path.join(script_dir, 'telco_churn_preprocessing', 'X_test.csv')

    try:
        X_test = pd.read_csv(X_test_data_path) 
    except FileNotFoundError:
        print(f"Error: File X_test.csv tidak ditemukan di {X_test_data_path}")
        print("Pastikan Anda sudah menjalankan preprocessing dan data ada di sana.")
        exit(1)

    # === Ambil satu sample baris data ===
    sample = X_test.iloc[[0]] 

    # === Setup Endpoint dan Header ===
    url = "http://127.0.0.1:1234/invocations"
    headers = {"Content-Type": "application/json"}

    # === Kirim request POST dan Tampilkan hasil prediksi ===
    print(f"Mengirim permintaan prediksi ke {url}...")
    try:
        response_json = make_prediction_request(sample, url, headers)
        print("Response:", response_json)
    except requests.exceptions.ConnectionError:
        print(f"Error: Tidak dapat terhubung ke server prediksi di {url}.")
        print("Pastikan model_serving_app.py sedang berjalan di port 1234.")
    except Exception as e:
        print(f"Terjadi kesalahan saat membuat permintaan: {e}")
