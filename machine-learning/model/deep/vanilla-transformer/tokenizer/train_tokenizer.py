from argparse import ArgumentParser
import pandas as pd 
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import Whitespace

START_TOK = "[CLS]"
END_TOK = "[SEP]"
OOV_TOKEN = "[UNK]"
PADDING_TOKEN = "[PAD]"

NUM_TOKENS = 10_0000
MAXLEN = 10_000

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file", help="location of the file to be used for tokenizer training")
    parser.add_argument("--output_file", help="location where to save the trained tokenizer")
    args = parser.parse_args()

    print("Read File")
    df = pd.read_csv(args.input_file, sep=';')
    reviews = df.text.tolist()

    print("Train Tokenizer")
    tokenizer = Tokenizer(BPE(unk_token=OOV_TOKEN))
    trainer = BpeTrainer(vocab_size=NUM_TOKENS, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    print(re)

    tokenizer.train_from_iterator( reviews , trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            (START_TOK, tokenizer.token_to_id(START_TOK)),
            (END_TOK, tokenizer.token_to_id(END_TOK)),
        ],
    )
    tokenizer.enable_padding(pad_id=3, pad_token=PADDING_TOKEN, length=MAXLEN)

    print("Save Tokenizer")
    tokenizer.save(args.output_file)