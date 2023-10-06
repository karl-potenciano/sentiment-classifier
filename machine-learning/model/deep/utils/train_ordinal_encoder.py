from sklearn.preprocessing import OrdinalEncoder
from argparse import ArgumentParser
import pandas as pd 
import pickle
import numpy as np


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", help="location of the file to be used for tokenizer training")
    parser.add_argument("--output_file", help="location where to save the trained tokenizer")
    args = parser.parse_args()

    print("Read File")
    df = pd.read_csv(args.input_file, sep=';')

    labels = df['expected_sentiment'].unique().tolist()

    print("Train Encoder")
    encoder = OrdinalEncoder(dtype=np.int32)
    encoder.fit(np.array(labels).reshape(-1,1))

    print("Save Encoder")
    with open(args.output_file, 'wb') as handle:
        pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
