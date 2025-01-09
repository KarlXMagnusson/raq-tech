from pathlib import Path
from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol defining the interface for LLM clients.
    
    This protocol ensures all LLM clients implement a (semi) consistent interface for image-to-text generation.
    """
    
    model_name: str

    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Generate content using the LLM model with text + image as input.

        :param prompt: A textual prompt to provide to the model.
        :param image_path: File path (string or Path) to an image to be included in the request.
        :return: The generated response text from the model.
        :raises FileNotFoundError: If the specified image_path does not exist.
        :raises Exception: If the underlying model call fails or an unexpected error occurs.
        """
        pass