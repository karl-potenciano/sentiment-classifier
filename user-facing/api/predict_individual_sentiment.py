from argparse import ArgumentParser, ArgumentTypeError, Namespace
from prediction_utils import init_model, predict_individual


def main_process(args: Namespace): 
    print("Initialize Model")
    model = init_model()

    print("Run Prediction")
    prediction = predict_individual(args.input_string, model)

    print(f"User Input: {args.input_string}")
    print(f"Model Output: {prediction['model_output']}")
    print(f"Confidence: {prediction['confidence']}")
    # print(prediction)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_string", help="String to be classified")
    args = parser.parse_args()
    main_process(args)

