# MLProject/inference.py
import mlflow
import pandas as pd
import logging
import joblib

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        """
        This method is called once when the model is loaded.
        It loads the actual model artifact.
        """
        try:
            self.model = mlflow.pyfunc.load_model(context.artifacts["model_path"])
            logging.info("Model loaded successfully from artifacts using mlflow.pyfunc.load_model.")
        except Exception as e:
            logging.error(f"Error loading model with mlflow.pyfunc.load_model: {e}")
            raise

    def predict(self, context, model_input):
        """
        This method is called for each prediction request.
        It takes input data and returns predictions.
        """
        if not isinstance(model_input, pd.DataFrame):
            logging.error("Input is not a Pandas DataFrame.")
            raise TypeError("Input must be a Pandas DataFrame.")
        
        try:
            predictions = self.model.predict(model_input)
            logging.info(f"Prediction made for {len(model_input)} samples.")
            return predictions
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

if __name__ == "__main__":
    # Contoh penggunaan lokal (bukan untuk serving, tapi untuk testing)
    # Bagian ini TIDAK AKAN dijalankan oleh 'mlflow models serve'
    # Jika ingin menguji secara lokal, Anda bisa uncomment ini dan sesuaikan jalur/data dummy
    
    # model_path = "MLProject/best_logistic_regression_model_artifact" # Pastikan ini path ke folder artifact Anda
    # loaded_model = mlflow.pyfunc.load_model(model_path)
    # dummy_input = pd.DataFrame(
    #     [[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
    #     columns=[f'feature_{i}' for i in range(20)] # Sesuaikan dengan 20 fitur dummy Anda
    # )
    # prediction = loaded_model.predict(dummy_input)
    # print(f"Dummy prediction: {prediction}")
    logging.info("inference.py script executed (designed for mlflow.pyfunc.PythonModel).")
