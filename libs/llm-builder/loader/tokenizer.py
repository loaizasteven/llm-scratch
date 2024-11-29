# Method to tokenize a input text

import re

from typing import Optional, Dict, Union
from pydantic import BaseModel

from functools import lru_cache


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
        preprocessed = self.preprocessing(text)

        tokens = sorted(set(preprocessed))
        self.token2id = {token:integer for integer, token in enumerate(tokens)}
        self.id2token = self._reverse_dict(self.token2id)

    def preprocessing(self, text: str):
        preprocessed = re.split(self.pattern, text)

        return [item.strip() for item in preprocessed if item.strip()]

    def encode(self, text:str) -> list:
        preprocessed = self.preprocessing(text)
        if not self.token2id:
            self._initialize_vocab(text)
            return self.token2id

        return [self.token2id[token] for token in preprocessed]
    
    def decode(self, ids: list) -> str:
        assert isinstance(ids, list), "Input should be a list"

        text = " ".join([self.id2token[id] for id in ids])
        processed = re.sub(r'\s+ ([,.?!"()\'])', r'\1', text)   # remove spaces before punctuation
        return processed
