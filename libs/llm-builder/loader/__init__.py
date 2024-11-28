from pathlib import Path

from pydantic import BaseModel
import sys

_THIS_DIR = Path(__file__).parent

class Commons(BaseModel):
    fileDir: str = str(_THIS_DIR)

sys.path.insert(0, Commons().fileDir)
