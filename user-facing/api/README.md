
# SAIL - Sentiment Classifier

This Folder contains the necessary scripts to run predictions both via command and and via API.

## Getting Started
### Set up 
1. Run `pip install -r requirements.txt`

### Running the Individual Sentiment Prediction Script
To produce a prediction simply run the following script 
```
python3 predict_individual_sentiment.py --input_string "INSERT INPUT STRING HERE"
```
Once execution is successful, you should be expecting an output similar to the one below
```
Initialize Model
Run Prediction
User Input: INSERT INPUT STRING HERE
Model Output: neutral
Confidence: 85.42
```


### Running the File Sentiment Prediction Script
To produce a prediction via file simply run the following script 
```
python3 predict_individual_sentiment.py --input_file <file_path_here> --output_file <file_path_here> --compute_metrics <True or False> --metrics_location <file_path_here>
```
Once execution is successful, you should be expecting an output similar to the one below
```
Initialize Model
Run Prediction
Save Predictions
```
The predictions should be saved on the supplied `output_file` path.

If `compute_metrics` was set to `True`, the following additional output should be expected
```
              precision  recall  f1-score  support
negative          94.16   81.92     87.61   177.00
neutral           81.21   87.05     84.03   139.00
positive          85.13   91.21     88.06   182.00
accuracy          86.75   86.75     86.75    86.75
macro avg         86.83   86.73     86.57   498.00
weighted avg      87.24   86.75     86.78   498.00
```

### Running the Flask API

To run the Flask API, simply execute the following command
```
flask --app sentiment_prediction_api run --host <host> --port <port>
```

#### Flask API Individual Prediction
An individual prediction endpoint can be found on `GET predict_sentiment/individual`. Sample Request and response can be found [here](https://github.com/karl-potenciano/sentiment-classifier/edit/main/user-facing/api/Sentiment Prediction API.postman_collection.json)

#### Flask API DataFrame Prediction
An individual prediction endpoint can be found on `POST predict_sentiment/dataframe`. Sample Request and response can be found [here](https://github.com/karl-potenciano/sentiment-classifier/edit/main/user-facing/api/Sentiment Prediction API.postman_collection.json)
