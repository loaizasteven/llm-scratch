from bs4 import BeautifulSoup

import urllib 
from urllib import request
from urllib import error
import os

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
