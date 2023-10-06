from argparse import ArgumentParser
from datasets import load_dataset, Dataset
import numpy as np
import pandas as pd
from transformers import XLMRobertaTokenizer, BertForSequenceClassification,BertConfig,TrainingArguments, Trainer,TextClassificationPipeline
import evaluate
import mlflow
import pickle
from sklearn.preprocessing import OrdinalEncoder
from typing import Tuple
from sklearn.metrics import classification_report

BASE_MODEL = "microsoft/Multilingual-MiniLM-L12-H384"
MAX_LENGTH = 74

def get_label_to_id_dicts(encoder_path: str) -> Tuple[OrdinalEncoder, dict, dict]:
    with open(encoder_path, "rb") as handle:
        encoder = pickle.load(handle)
    
    label2id = dict()
    id2label = dict()

    for idx, label in enumerate(encoder.categories_[0]):
        label2id[label] = idx 
        id2label[idx] = label 

    return encoder, label2id, id2label

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

def prepare_dataset(path: str, encoder: OrdinalEncoder, tokenizer: XLMRobertaTokenizer, sep=None) -> Dataset:
    df = pd.read_csv(path, sep=sep)
    df = df[df['text'].notna()]
    df['label'] = encoder.transform(np.array(df.expected_sentiment).reshape(-1,1))
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    tokenized_dataset = dataset.map(lambda example: tokenize_function(example, tokenizer), batched=True)
    return tokenized_dataset

def init_model(label2id: dict, id2label: dict) -> Tuple[XLMRobertaTokenizer, BertForSequenceClassification]:
    tokenizer = XLMRobertaTokenizer.from_pretrained(BASE_MODEL)
    num_labels = len(label2id.keys())

    config = BertConfig.from_pretrained(BASE_MODEL, num_labels=num_labels, id2label=id2label, label2id=label2id)
    model = BertForSequenceClassification.from_pretrained(BASE_MODEL,config=config)

    return tokenizer, model


def init_trainer(model_output_path: str, model: BertForSequenceClassification, train_dataset: Dataset, val_dataset:Dataset) -> Trainer:
    training_args = TrainingArguments(output_dir=model_output_path, evaluation_strategy="epoch", num_train_epochs=5, use_mps_device=True, save_strategy='no', per_device_train_batch_size=256, per_device_eval_batch_size=256)
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

    print("Get encoders and labels")
    encoder, label2id, id2label = get_label_to_id_dicts(args.ordinal_encoder)

    print("Init Tokenizer and Model")
    tokenizer, model = init_model(label2id, id2label)

    print("Get Datasets")
    train_dataset = prepare_dataset(args.train_dataset, encoder, tokenizer,';')
    val_dataset = prepare_dataset(args.val_dataset, encoder, tokenizer,';')
    

    print("Init Trainer")
    trainer = init_trainer(args.model_output_path, model, train_dataset, val_dataset)

    print("Train Model")
    trainer.train()
    # trainer.save_model(args.model_output_path)

    print("Evaluate against test set")
    test_dataset = pd.read_csv(args.test_dataset)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer,device="mps")
    pipe.save_pretrained(args.model_output_path)
    results_df = pd.DataFrame(pipe(test_dataset['text'].tolist()))

    print(classification_report(test_dataset.expected_sentiment, results_df.label))
    mlflow.end_run()