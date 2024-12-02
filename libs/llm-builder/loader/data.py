from bs4 import BeautifulSoup

import urllib 
from urllib import request
from urllib import error
import os
import json

from pydantic import BaseModel
from typing import Optional, Any, List

from torch.utils.data import Dataset


class GPTDatasetV1(BaseModel, Dataset):
        txt: str =''
        tokenizer: Any = None
        maxLength: Optional[int] = 1024
        stride: Optional[int] = 512
        inputIds: List[List[int]] = []
        targetIds: List[List[int]] = []

        def __init__(self, **data):
            super().__init__(**data)
            token_ids = self.tokenizer.encode(self.txt)

            for i in range(0, len(token_ids) - data["maxLength"] + 1, data["stride"]):
                self.inputIds.append(token_ids[i:i + data["maxLength"]])
                self.targetIds.append(token_ids[i + 1:i + data["maxLength"] + 1])

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
        

def htmlParser(text: bytes) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    content = soup.find('div', class_='mw-parser-output')
    return content.get_text()


def download_text(url, filePath, htmlParser=False):
    try:
        if htmlParser:
            response = request.urlopen(url)
            content = response.read()
            with open(filePath, 'wb', encoding='utf-8') as file:
                file.write(content)
        else:
            os.makedirs(os.path.dirname(filePath), exist_ok=True)
            urllib.request.urlretrieve(url, filePath)
    except (error.URLError, error.HTTPError) as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import os.path as osp
    from loader import Commons

    # Load the text data from the URL (original source "https://en.wikisource.org/wiki/The_Verdict")
    URL_ = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
    WRITE_PATH_ = osp.join(Commons().parentDir, 'scratchpad/data/the-verdict.txt')

    download_text(URL_, WRITE_PATH_)
