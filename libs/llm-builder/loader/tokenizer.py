# Method to tokenize a input text
"""
This module provides a simple tokenizer for text processing.
It includes methods for tokenizing text, encoding text into token IDs,
and decoding token IDs back into text. The tokenizer uses regular expressions
to split text into tokens based on a specified pattern.

Modifications:
* Pydantic BaseModel is used to define the class.
* SimpleTokenizer class handled vocab initialization and token encoding.
* SimpleTokenizerV2 inherits SimpleTokenizer and adds special tokens.
"""

import re
import json
import os.path as osp

from typing import Optional, List, Dict, Union
from pydantic import BaseModel

from functools import lru_cache

@lru_cache(maxsize=1500)
def preprocessing(text: str, pattern: str):
    preprocessed = re.split(pattern, text)

    return [item.strip() for item in preprocessed if item.strip()]


class SimpleTokenizer(BaseModel):
    token2id: Optional[Dict] = None
    id2token: Optional[Dict] = None
    pattern: str = r'([,.:?_!"()\']|--|\s)'

    def __init__(self, token2id: Optional[Dict] = None, id2token: Optional[Dict] = None, pattern: str = r'([,.:?_!"()\']|--|\s)'):
        super().__init__()
        self.token2id = token2id
        self.pattern = pattern
        self.id2token = self._reverse_dict(token2id)

    @staticmethod
    def _reverse_dict(dictionary: Union[Dict, None]) -> Dict:
        if dictionary:
            return {v:k for k,v in dictionary.items()}
        
    def _initialize_vocab(self, text: str):
        preprocessed = preprocessing(text, self.pattern)

        tokens = sorted(set(preprocessed))
        self.token2id = {token:integer for integer, token in enumerate(tokens)}
        self.id2token = self._reverse_dict(self.token2id)

    def encode(self, text:str) -> list:
        preprocessed = preprocessing(text, self.pattern)
        if not self.token2id:
            self._initialize_vocab(text)
            return self.token2id

        return [self.token2id[token] for token in preprocessed]
    
    def decode(self, ids: list) -> str:
        assert isinstance(ids, list), "Input should be a list"

        text = " ".join([self.id2token[id] for id in ids])
        processed = re.sub(r'\s+ ([,.?!"()\'])', r'\1', text)   # remove spaces before punctuation
        return processed


class SimpleTokenizerV2(SimpleTokenizer):
    specialTokens: List[str] = ['<endoftext>', '<unk>']
    
    def _initialize_vocab(self, text):
        preprocessed = preprocessing(text, self.pattern)

        tokens = sorted(set(preprocessed))
        # extend the tokens with special tokens
        tokens.extend(self.specialTokens)

        self.token2id = {token:integer for integer, token in enumerate(tokens)}
        self.id2token = self._reverse_dict(self.token2id)

    def encode(self, text:str) -> list:
        preprocessed = preprocessing(text, self.pattern)
        unknownId = self.token2id.get('<unk>', -1)

        if not self.token2id:
            self._initialize_vocab(text)
            return self.token2id

        return [self.token2id.get(token, unknownId) for token in preprocessed]

    def dump_vocab(self, path: str):
        dir = osp.dirname(path)
        for attr, obj in self.__dict__.items():
            if isinstance(obj, dict):
                savePath = osp.join(dir, f"{attr}.json")
                json.dump(obj, open(savePath, 'w'), indent=4)  
