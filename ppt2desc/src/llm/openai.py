import os
import base64
from pathlib import Path
from typing import Optional, Union
import logging

from openai import OpenAI

# Remove OpenAI's standard logging messages
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

class OpenAIClient:
    """
    A client wrapper around OpenAI's API for image + prompt generation.

    Usage:
        client = OpenAIClient(api_key="YOUR_KEY", model="gpt-4o")
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialize the OpenAI client with API key and model name.

        :param api_key: Optional API key string. If not provided,
                       checks the OPENAI_API_KEY environment variable.
        :param model: The name of the generative model to use (e.g. "gpt-4-vision-preview").
        :raises ValueError: If no API key is found or model is None.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via OPENAI_API_KEY environment variable."
            )

        if model is None:
            raise ValueError("The 'model' argument is required and cannot be None.")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64 string.

        :param image_path: Path to the image file
        :return: Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Generate content using the OpenAI model with text + image as input.

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
            # Encode the image to base64
            base64_image = self._encode_image(image_path_obj)

            # Create the API request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Failed to generate content with OpenAI model: {e}")