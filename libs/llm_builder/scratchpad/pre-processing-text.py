from pathlib import Path

import sys
import os

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent

sys.path.insert(0, str(_PARENT_DIR))
from loader import data, tokenizer

# Load the text data from the URL
URL_ = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")   
READ_PATH_ = WRITE_PATH_ = _THIS_DIR / 'data/the-verdict.txt'

if os.path.exists(READ_PATH_):
    print(f"File already exists: {READ_PATH_}")
else:
    data.download_text(URL_, WRITE_PATH_)

with open(READ_PATH_, 'r') as file:
        text = file.read()


if __name__ == "__main__":
    import argparse
    from data import GPTDatasetV1, CustomDataLoader
    import tiktoken

    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1)   
    args = parser.parse_args()

    if args.version == 1:
        tokenizer = tokenizer.SimpleTokenizerV2()
        tokenizer._initialize_vocab(text)

        input_ = "Hi, I found the Brown Fox!"
        encoded = tokenizer.encode(input_)
        tokenizer.dump_vocab(_THIS_DIR / 'data/vocab.json')

        print(input_)
        print(tokenizer.decode(encoded))
    else:
        dataloader = CustomDataLoader(
            txt=text, 
            dataset='GPTDatasetV1',
            batch_size=1,
            max_length=4,
            stride=1, 
            shuffle=False
        ).loader()

        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print(f"Input \n: {inputs}")
        print(f"Target \n: {targets}")
