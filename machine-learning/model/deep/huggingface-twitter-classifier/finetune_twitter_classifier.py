from argparse import ArgumentParser
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig,TrainingArguments, Trainer,TextClassificationPipeline
import evaluate
import mlflow
import pickle
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple
from sklearn.metrics import classification_report

BASE_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH = 31

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def prepare_dataset(path: str, label2id:dict, tokenizer: AutoTokenizer, sep=None) -> Dataset:
    df = pd.read_csv(path, sep=sep)
    df.columns = ['text', 'label']
    df['label'] = df.label.apply(lambda label: label2id.get(label))
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    tokenized_dataset = dataset.map(lambda example: tokenize_function(example, tokenizer), batched=True)
    return tokenized_dataset

def init_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, dict]:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    config = AutoConfig.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL,config=config)

    return tokenizer, model, config.label2id


def init_trainer(model_output_path: str, model: AutoModelForSequenceClassification, train_dataset: Dataset, val_dataset:Dataset) -> Trainer:
    training_args = TrainingArguments(output_dir=model_output_path, evaluation_strategy="epoch", num_train_epochs=10, use_mps_device=True, save_strategy='no', per_device_train_batch_size=128, per_device_eval_batch_size=128)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    return trainer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_dataset")
    parser.add_argument("--val_dataset")
    parser.add_argument("--test_dataset")
    parser.add_argument("--model_output_path")
    parser.add_argument("--device")
    parser.add_argument("--ordinal_encoder")

    args = parser.parse_args()

    print("Init Tokenizer and Model")
    tokenizer, model, label2id = init_model()

    print("Get Datasets")
    train_dataset = prepare_dataset(args.train_dataset, label2id, tokenizer,';')
    val_dataset = prepare_dataset(args.val_dataset, label2id, tokenizer,';')
    

    print("Init Trainer")
    trainer = init_trainer(args.model_output_path, model, train_dataset, val_dataset)

    print("Train Model")
    trainer.train()
    trainer.save_model(args.model_output_path)

    print("Evaluate against test set")
    test_dataset = pd.read_csv(args.test_dataset)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer,device="mps")

    results_df = pd.DataFrame(pipe(test_dataset['text'].tolist()))

    print(classification_report(test_dataset.expected_sentiment, results_df.label))
    mlflow.end_run()