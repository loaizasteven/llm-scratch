from bs4 import BeautifulSoup

import urllib 
from urllib import request
from urllib import error
import os

from pydantic import BaseModel, ConfigDict
from typing import Optional, Any, List

from torch.utils.data import Dataset, DataLoader

import tiktoken


class GPTDatasetV1(BaseModel, Dataset):
        txt: str
        tokenizer: Any
        max_length: Optional[int] = 1024
        stride: Optional[int] = 512
        input_ids: List[List[int]] = []
        target_ids: List[List[int]] = []

        def __init__(self, **data):
            super().__init__(**data)
            self._initialize()

        def _initialize(self):
            token_ids = self.tokenizer.encode(self.txt)
            for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                self.input_ids.append(token_ids[i:i + self.max_length])
                self.target_ids.append(token_ids[i + 1:i + self.max_length + 1])
                
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]


class  CustomDataLoader(BaseModel):
    txt: str
    dataset: str
    dataset_mapping: dict = {'GPTDatasetV1': GPTDatasetV1}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    batch_size: int = 2
    max_length: int = 256
    stride: int = 128
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0

    
    def custom_collate_fn(batch):
        input_ids = [torch.tensor(item[0]) for item in batch]
        target_ids = [torch.tensor(item[1]) for item in batch]
        
        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)
        
        return input_ids, target_ids

    def loader(self, encoder='gpt2'):
        tokenizer = tiktoken.get_encoding(encoder)
        dataset = self.dataset_mapping[self.dataset]
        dataset = dataset(txt=self.txt, tokenizer=tokenizer, max_length=self.max_length, stride=self.stride, collate_fn=self.custom_collate_fn)
        
        return DataLoader(
                    dataset, 
                    batch_size=self.batch_size, 
                    shuffle=self.shuffle, 
                    drop_last=self.drop_last, 
                    num_workers=self.num_workers
                )


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
