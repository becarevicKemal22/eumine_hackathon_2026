from abc import ABC, abstractmethod
from typing import Dict, List

from pymatgen.core import Structure


class MatFedPredictor(ABC):
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, structures: List[Structure]) -> List[Dict]:
        pass

    @abstractmethod
    def describe(self) -> Dict:
        pass
