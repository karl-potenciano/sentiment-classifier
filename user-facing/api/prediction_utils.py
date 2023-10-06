from transformers import pipeline, TextClassificationPipeline
import pandas as pd
from sklearn.metrics import classification_report
from typing import Literal, Union

MODEL_PATH = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def init_model() -> TextClassificationPipeline:
    model = pipeline('sentiment-analysis', model=MODEL_PATH, tokenizer=MODEL_PATH)
    return model

def predict_dataframe(df: pd.DataFrame, model: TextClassificationPipeline) -> pd.DataFrame:
    prediction_list = model(df.text.tolist())
    prediction_df = pd.DataFrame(prediction_list)
    prediction_df.columns = ["model_output", "confidence"]
    prediction_df["confidence"] = prediction_df.confidence.apply(lambda conf: round(conf * 100.,2))

    return prediction_df

def compute_metrics(df: pd.DataFrame, format=Literal["pandas", "dict"]) -> Union[pd.DataFrame, dict]:
    report = classification_report(df.expected_sentiment, df.model_output, output_dict=True)
    final_report = report

    convert_to_100 = ["precision", "recall", "f1-score"]
    metrics = convert_to_100 + ["support"]
    for key in final_report.keys():
        if type(final_report[key]) == float:
            final_report[key] = round(final_report[key] * 100, 2)
        else:
            for metric in metrics:
                final_report[key][metric] = round(final_report[key][metric] * 100,2) if metric in convert_to_100 else round(final_report[key][metric],0)
    if format == "pandas":
        final_report = pd.DataFrame(report).transpose()
    
    return final_report


def predict_individual(input_string: str, model: TextClassificationPipeline) -> dict:
    prediction = model(input_string)[0]
    formatted_prediction = {"model_output": prediction['label'], "confidence": round(prediction['score'] * 100., 2)}

    return formatted_prediction