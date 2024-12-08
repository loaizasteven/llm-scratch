from pathlib import Path

from pydantic import BaseModel
import sys

_THIS_DIR = Path(__file__).parent
_PARENT_DIR = _THIS_DIR.parent

class Commons(BaseModel):
    fileDir: str = str(_THIS_DIR)
    parentDir: str = str(_PARENT_DIR)

sys.path.insert(0, Commons().fileDir)
sys.path.insert(0, Commons().parentDir)
