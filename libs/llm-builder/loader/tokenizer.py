# Method to tokenize a input text

import re

def basic_text_to_word(text):
    # split on punctuation, commas, and whitespace
    pattern = r'([,.:?_!"()\']|--|\s)'
    textls = re.split(pattern, text)

    # Tokenize remove white space
    return [item.strip() for item in textls if item.strip()]
