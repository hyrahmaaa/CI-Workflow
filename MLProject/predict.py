import requests
import pandas as pd
import json

# === Load preprocessed CSV untuk inference ===
X_test = pd.read_csv('./MLProject/telco_churn_preprocessing/X_test.csv')  # pastikan path dan nama file benar

# === Ambil satu sample baris data ===
sample = X_test.iloc[[0]]  # [[]] agar tetap dalam bentuk DataFrame

# === Format payload sesuai MLflow scoring API (v2.0+) ===
payload = {
    "inputs": sample.to_dict(orient="records")
}

# === Setup Endpoint dan Header ===
url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}

# === Kirim request POST ===
response = requests.post(url, headers=headers, data=json.dumps(payload))

# === Tampilkan hasil prediksi ===
print("Response:", response.json())
