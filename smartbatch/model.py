from abc import ABC, abstractmethod
from typing import List, Any

class BaseModel(ABC):
    """
    Abstract base class for models served by SmartBatch.
    Users are not required to inherit from this, but it serves as a good template.
    Protocol: Callable[[List[Any]], List[Any]]
    """
    
    @abstractmethod
    def infer(self, batch: List[Any]) -> List[Any]:
        """
        Takes a list of inputs and returns a list of outputs of the same length.
        """
        pass
