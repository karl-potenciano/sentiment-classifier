from argparse import ArgumentParser
import pandas as pd
import os, sys
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_folder", help="location of folder to be processed")
    parser.add_argument("--output_folder", help="location of folder where to store processed datasets")
    args = parser.parse_args()

    print('Reading Files')
    filenames = os.listdir(args.input_folder)
    dataframes = []

    for filename in filenames:
        filepath = os.path.join(args.input_folder, filename)
        df = pd.read_csv(filepath, sep=';')
        dataframes.append(df)

    print("Combining Dataframes")
    final_df = pd.concat(dataframes)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    
    print("Saving master file")
    output_file = os.path.join(args.output_folder, "reviews.csv")
    final_df.to_csv(output_file, index=False, encoding='utf-8', sep=';')

    train_df, val_df = train_test_split(final_df, test_size=0.25,stratify=final_df['expected_sentiment'])

    print("Saving Train Set")
    train_file_path = os.path.join(args.output_folder, "train.csv")
    train_df.to_csv(train_file_path, index=False, encoding='utf-8', sep=';')

    print("Saving Validation Set")
    val_file_path = os.path.join(args.output_folder, "val.csv")
    val_df.to_csv(val_file_path, index=False, encoding='utf-8', sep=';')