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

# Tokenize the text
with open(READ_PATH_, 'r') as file:
    text = file.read()

tokens = tokenizer.text2word(text)
print(tokens[:10])