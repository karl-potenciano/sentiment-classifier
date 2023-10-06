from flask import Flask, request, g, jsonify
from prediction_utils import init_model, predict_individual, predict_dataframe, compute_metrics
import pandas as pd
import json

model = init_model()


app = Flask(__name__)

@app.route("/predict_sentiment/individual", methods=["GET"])
def predict_sentiment_individual():
    args = request.args
    status_code = 200
    if "input_string" in args and args["input_string"] is not None and args["input_string"].strip() != '':
        prediction = predict_individual(args["input_string"], model)
        prediction["input_string"] = args["input_string"]
    else:
        prediction = {"error": True, "message": "Missing Input"}
        status_code = 400
    
    return jsonify(prediction), status_code


@app.route("/predict_sentiment/dataframe", methods=["POST"])
def predict_sentiment_dataframe():
    args = request.form

    df = pd.read_csv(request.files.get("input_file"))
    predictions_df = predict_dataframe(df, model)
    final_df = df.merge(predictions_df, left_index=True, right_index=True)
    predictions_json = final_df.to_dict(orient="records")

    return_json = {"results": predictions_json}    

    if args["compute_metrics"]:
        metrics = compute_metrics(final_df, "dict")
        return_json["metrics"] = metrics
    return jsonify(return_json) 