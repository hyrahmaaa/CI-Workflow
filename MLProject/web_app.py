from flask import Flask, render_template, request
import pandas as pd
import requests

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        input_data = {}
        for field in request.form:
            input_data[field] = float(request.form[field])
        
        df = pd.DataFrame([input_data])
        
        response = requests.post(
            "http://127.0.0.1:1234/invocations",
            headers={"Content-Type": "application/json"},
            json={"dataframe_split": df.to_dict(orient="split")}
        )
        prediction = response.json()["predictions"][0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
