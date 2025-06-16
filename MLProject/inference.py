import mlflow.pyfunc
import joblib

class ChurnPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load model dari artifacts
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        # model_input biasanya DataFrame
        return self.model.predict(model_input)
