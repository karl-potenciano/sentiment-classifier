from argparse import ArgumentParser, ArgumentTypeError, Namespace
from prediction_utils import init_model,predict_dataframe,compute_metrics
from transformers import TextClassificationPipeline
import pandas as pd
from sklearn.metrics import classification_report



def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def predict_sentiment(df: pd.DataFrame, model: TextClassificationPipeline) -> pd.DataFrame:
    prediction_df = predict_dataframe(df, model)
    final_df = df.merge(prediction_df, left_index=True, right_index=True)
    
    return final_df

def main_process(args: Namespace): 
    print("Initialize Model")
    model = init_model()

    print("Read Input File")
    df = pd.read_csv(args.input_file)

    print("Run Prediction")
    predicted_df = predict_sentiment(df, model)

    print("Save Predictions")
    predicted_df.to_csv(args.output_file)

    if args.compute_metrics:
        print("Computing Metrics")
        metrics = compute_metrics(predicted_df, "pandas")
        print(metrics)

        if args.metrics_location is not None:
            metrics.to_csv(args.metrics_location)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", help="Location of the file to be processed. File should have column `text` as input")
    parser.add_argument("--output_file", help="Location where to store the predictions")
    parser.add_argument("--compute_metrics", help="Flag used to control metrics computation. File should have column `expected_sentiment` as expected label",type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument("--metrics_location", default=None, help="Location where to store metrics when necessary")

    args = parser.parse_args()


    main_process(args)

