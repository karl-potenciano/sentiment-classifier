from argparse import ArgumentParser
import json
import pandas as pd

OUTLIER_SAMPLING_RATE = 0.05
NON_OUTLIER_SAMPLING_RATE = 0.20



SENTIMENT_LABELS = {
    5.0: 'positive',
    4.0: 'positive',
    3.0: 'neutral',
    2.0: 'negative',
    1.0: 'negative',
}

RELEVANT_FIELDS = ['overall', 'reviewText']

def build_dataframe(filepath: str) -> pd.DataFrame:
    file = open(filepath, "r")
    review_lines = file.readlines()
    file.close()

    processed_review_lines = [json.loads(line.strip()) for line in review_lines]

    reviews_df = pd.DataFrame(processed_review_lines)
    reviews_df['review_length'] = reviews_df.reviewText.str.len()

    return reviews_df

def transform_dataset_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df[RELEVANT_FIELDS]
    df.drop_duplicates(inplace=True)
    df['expected_sentiment'] = df.overall.apply(lambda rating: SENTIMENT_LABELS.get(rating))
    df.rename({'reviewText': 'text'},axis=1,inplace=True)
    
    df.drop('overall', axis=1, inplace=True)
    
    return df

def sample_dataframe(df: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    reference_count = df.expected_sentiment.value_counts().min()
    sample_count = round(reference_count * sample_rate)

    sampled_df = df.groupby('expected_sentiment', group_keys=False).apply(lambda label: label.sample(sample_count))

    return sampled_df


def build_dataset_for_modeling(df: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    transformed_df = transform_dataset_labels(df)
    sampled_df = sample_dataframe(transformed_df, sample_rate)

    return sampled_df



def process_amazon_reviews(input_file: str, output_file: str):
    print("Loading Reviews")
    reviews_df = build_dataframe(input_file)
    
    print("Filtering Verified Reviews")
    verified_reviews_df = reviews_df[reviews_df.verified]
    quartile = verified_reviews_df.review_length.quantile([0.25, 0.75])

    print("Building dataset within IQR")
    iqr_df = build_dataset_for_modeling(verified_reviews_df[(verified_reviews_df.review_length >= quartile[0.25]) & (verified_reviews_df.review_length <= quartile[0.75])], NON_OUTLIER_SAMPLING_RATE)

    print("Building dataset outside IQR")
    outlier_df = build_dataset_for_modeling(verified_reviews_df[(verified_reviews_df.review_length <= quartile[0.25]) | (verified_reviews_df.review_length >= quartile[0.75])], OUTLIER_SAMPLING_RATE)
    
    print("Combining datasets")
    final_df = pd.concat([iqr_df, outlier_df])
    
    print("Saving to file")
    final_df.to_csv(output_file, encoding="utf-8", index=False, sep=';')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", help="location of file to be processed")
    parser.add_argument("--output_file", help="location where to save the processed dataset")
    args = parser.parse_args()

    process_amazon_reviews(args.input_file, args.output_file)