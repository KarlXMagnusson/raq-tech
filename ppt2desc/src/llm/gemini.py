import os
from pathlib import Path
from typing import Optional, Union

import PIL.Image
import google.generativeai as genai


class GeminiClient:
    """
    A client wrapper around Google's Generative AI (Gemini) model for image + prompt generation.

    Usage:
        client = GeminiClient(api_key="YOUR_KEY", model="gemini-1.5-flash")
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialize the Gemini client with API key and model name.

        :param api_key: Optional API key string. If not provided,
                        checks the GEMINI_API_KEY environment variable.
        :param model:   The name of the generative model to use (e.g. "gemini-1.5-flash").
        :raises ValueError: If no API key is found or model is None.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set via GEMINI_API_KEY environment variable.")

        if model is None:
            raise ValueError("The 'model' argument is required and cannot be None.")

        # Configure generative AI
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Generate content using the Gemini model with text + image as input.

        :param prompt: A textual prompt to provide to the model.
        :param image_path: File path (string or Path) to an image to be included in the request.
        :return: The generated response text from the model.
        :raises FileNotFoundError: If the specified image_path does not exist.
        :raises Exception: If the underlying model call fails or an unexpected error occurs.
        """
        # Ensure the image path exists
        image_path_obj = Path(image_path)
        if not image_path_obj.is_file():
            raise FileNotFoundError(f"Image file not found at {image_path_obj}")

        try:
            image = PIL.Image.open(image_path_obj)
            # If using the google.generativeai library's generate_content method:
            # pass [prompt, image] in the format required by the library
            response = self.model.generate_content([prompt, image])
            return response.text

        except Exception as e:
            raise Exception(f"Failed to generate content with Gemini model: {e}")
