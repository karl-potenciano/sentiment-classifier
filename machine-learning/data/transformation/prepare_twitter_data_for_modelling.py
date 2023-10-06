from argparse import ArgumentParser
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split

LABELS = {-1.0: 'negative', 0: 'neutral', 1.0: 'positive'}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", help="location of folder to be processed")
    parser.add_argument("--output_folder", help="location of folder where to store processed datasets")
    args = parser.parse_args()

    print('Reading Files')
    df = pd.read_csv(args.input_file)
    df = df[df.category.notna()]
    df['expected_sentiment'] = df.category.apply(lambda row: LABELS.get(row))
    df.columns = ['text', '_', 'expected_sentiment']
    
    final_df = df[['text', 'expected_sentiment']]
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    print("Saving master file")
    output_file = os.path.join(args.output_folder, "tweets.csv")
    final_df.to_csv(output_file, index=False, encoding='utf-8', sep=';')

    train_df, val_df = train_test_split(final_df, test_size=0.25,stratify=df['expected_sentiment'])

    print("Saving Train Set")
    train_file_path = os.path.join(args.output_folder, "train.csv")
    train_df.to_csv(train_file_path, index=False, encoding='utf-8', sep=';')

    print("Saving Validation Set")
    val_file_path = os.path.join(args.output_folder, "val.csv")
    val_df.to_csv(val_file_path, index=False, encoding='utf-8', sep=';')