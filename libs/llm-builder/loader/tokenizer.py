# Method to tokenize a input text

import re

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
