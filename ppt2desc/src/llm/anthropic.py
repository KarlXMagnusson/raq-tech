import os
import base64
from pathlib import Path
from typing import Optional, Union

import anthropic

class AnthropicClient:
    """
    A client wrapper around Anthropic's API for image + prompt generation.

    Usage:
        client = AnthropicClient(api_key="YOUR_KEY", model="claude-3-5-sonnet-latest")
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialize the Anthropic client with API key and model name.

        :param api_key: Optional API key string. If not provided,
                       checks the ANTHROPIC_API_KEY environment variable.
        :param model: The name of the generative model to use (e.g. "claude-3-sonnet-20240229").
        :raises ValueError: If no API key is found or model is None.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via ANTHROPIC_API_KEY environment variable."
            )

        if model is None:
            raise ValueError("The 'model' argument is required and cannot be None.")

        self.client = anthropic.Anthropic(api_key=self.api_key)
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
        Generate content using the Anthropic model with text + image as input.

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

            # Create the messages request
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=8192,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            return response.content[0].text

        except Exception as e:
            raise Exception(f"Failed to generate content with Anthropic model: {e}")