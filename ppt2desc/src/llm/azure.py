import os
import base64
from pathlib import Path
from typing import Optional, Union

from openai import AzureOpenAI


class AzureClient:
    """
    A client wrapper around Azure OpenAI's API for image + prompt generation.

    Usage:
        client = AzureClient(
            api_key="YOUR_KEY",
            endpoint="YOUR_ENDPOINT",
            deployment="deployment_name",
            api_version="2023-12-01-preview"
        )
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> None:
        """
        Initialize the Azure OpenAI client.

        :param api_key: Optional API key string. If not provided,
                       checks the AZURE_OPENAI_API_KEY environment variable.
        :param endpoint: Azure OpenAI endpoint. If not provided,
                        checks the AZURE_OPENAI_ENDPOINT environment variable.
        :param deployment: The deployment name for the model.
        :param api_version: Azure OpenAI API version (e.g., "2023-12-01-preview")
        :raises ValueError: If required parameters are missing.
        """
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set via AZURE_OPENAI_API_KEY environment variable."
            )

        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.endpoint:
            raise ValueError(
                "Endpoint must be provided or set via AZURE_OPENAI_ENDPOINT environment variable."
            )

        if deployment is None:
            raise ValueError("The 'deployment' argument is required and cannot be None.")

        if api_version is None:
            raise ValueError("The 'api_version' argument is required and cannot be None.")

        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=api_version,
            base_url=f"{self.endpoint}/openai/deployments/{deployment}"
        )
        self.deployment = deployment
        
        # For JSON metadata
        self.model_name = deployment

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
        Generate content using the Azure OpenAI model with text + image as input.

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
                model=self.deployment,
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
            raise Exception(f"Failed to generate content with Azure OpenAI model: {e}")