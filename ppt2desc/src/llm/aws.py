import os
from pathlib import Path
from typing import Optional, Union

import boto3
class AWSClient:
    """
    A client wrapper around AWS Bedrock Runtime API.

    Usage:
        client = AWSClient(
            access_key_id="YOUR_ACCESS_KEY",
            secret_access_key="YOUR_SECRET_KEY",
            region="us-east-1",
            model="amazon.nova-pro-v1:0"  # or any Claude model
        )
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the AWS Bedrock client.

        :param access_key_id: Optional AWS access key ID. If not provided,
                            checks AWS_ACCESS_KEY_ID environment variable.
        :param secret_access_key: Optional AWS secret access key. If not provided,
                                checks AWS_SECRET_ACCESS_KEY environment variable.
        :param region: AWS region name. If not provided, checks AWS_REGION environment variable.
        :param model: The model ID (e.g., "amazon.nova-pro-v1:0" or any Claude model).
        :raises ValueError: If required parameters are missing.
        """
        self.access_key_id = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        if not self.access_key_id:
            raise ValueError(
                "AWS access key ID must be provided or set via AWS_ACCESS_KEY_ID environment variable."
            )

        self.secret_access_key = secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not self.secret_access_key:
            raise ValueError(
                "AWS secret access key must be provided or set via AWS_SECRET_ACCESS_KEY environment variable."
            )

        self.region = region or os.environ.get("AWS_REGION")
        if not self.region:
            raise ValueError(
                "AWS region must be provided or set via AWS_REGION environment variable."
            )

        if model is None:
            raise ValueError("The 'model' argument is required and cannot be None.")

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=self.region,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key
        )
        self.model_id = model

        # For JSON metadata
        self.model_name = model

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file to base64 string.

        :param image_path: Path to the image file
        :return: Base64 encoded string of the image
        """
        with open(image_path, "rb") as image_file:
            return image_file.read()

    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Generate content using the AWS model with text + image as input.

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

            # Create the messages list
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        },
                        {
                            "image": {
                                "format": "png",
                                "source": {
                                    "bytes": base64_image
                                }
                            }
                        }
                    ]
                }
            ]

            # Invoke the model using converse
            response = self.client.converse(
                modelId=self.model_id,
                messages=messages
            )

            return response["output"]["message"]["content"][0]["text"]

        except Exception as e:
            raise Exception(f"Failed to generate content with AWS model: {e}")