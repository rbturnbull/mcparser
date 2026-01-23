from pathlib import Path
from collections import ABC

DEFAULT_NUM_RESULTS = 10

class DatabaseBase(ABC):
    def __init__(self, path:Path | str):
        self.path = Path(path)
        self.load(self.path)

    @classmethod
    def build(
        self,
        conllu_path:Path,
        output_path:Path,
    ) -> "DatabaseBase":
        raise NotImplementedError()
    
    def load(
        self,
        path:Path | str,
    ):
        raise NotImplementedError()
    
    def __call__(
        self,
        query:str,
        num_results:int = DEFAULT_NUM_RESULTS,
    ) -> list[dict]:
        raise NotImplementedError()