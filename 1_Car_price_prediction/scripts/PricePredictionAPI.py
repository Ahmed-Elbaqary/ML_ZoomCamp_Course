from Car_Salary_Prediction import *
from flask import Flask, request, jsonify

# ----------------------------------------------
MLFLOW_TRACKING_URI = "../models/mlruns"
MLFLOW_RUN_ID = "871e33a94e9a42819d93d28209cfca7e"

# ----------------------------------------------

""" Initialize API and JobPrediction object """
app = Flask(__name__)
model = PricePrediction(mlflow_uri=MLFLOW_TRACKING_URI,
                        run_id=MLFLOW_RUN_ID)


# Create Prediction endpoint
@app.route("/Predict_Car_Price", methods=["POST"])
def predict_car_price():
    car_features = request.get_json()
    predictions = model.predict_car_price(car_features)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(port=5000)
