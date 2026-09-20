from abc import ABC, abstractmethod


class LLM(ABC):
    """Interface that every AIL language model must implement."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        raise NotImplementedError