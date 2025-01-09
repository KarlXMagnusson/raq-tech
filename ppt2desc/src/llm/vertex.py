import os
from pathlib import Path
from typing import Optional, Union

import vertexai
from vertexai.preview.generative_models import GenerativeModel, Image


class VertexAIClient:
    """
    A client wrapper around Google's Vertex AI service for image + prompt generation using Gemini models.

    Usage:
        client = VertexAIClient(
            credentials_path="path/to/credentials.json",
            project_id="your-project-id",
            region="us-central1",
            model="gemini-1.5-pro-002"
        )
        text_response = client.generate(prompt="Hello World", image_path="path/to/image.png")
    """

    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the Vertex AI client with necessary credentials and configuration.

        :param credentials_path: Path to the service account credentials JSON file.
                               If not provided, checks GOOGLE_APPLICATION_CREDENTIALS env var.
        :param project_id: GCP project ID. If not provided, checks PROJECT_ID env var.
        :param region: GCP region for Vertex AI. If not provided, checks REGION env var.
        :param model: The name of the generative model to use (e.g. "gemini-1.5-pro-002").
        :raises ValueError: If required credentials or configuration are missing.
        """
        # Check credentials
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not self.credentials_path:
            raise ValueError(
                "Credentials path must be provided or set via "
                "GOOGLE_APPLICATION_CREDENTIALS environment variable."
            )
        if not Path(self.credentials_path).is_file():
            raise FileNotFoundError(
                f"Credentials file not found at {self.credentials_path}"
            )
        
        # Check project ID and region
        self.project_id = project_id or os.environ.get("PROJECT_ID")
        if not self.project_id:
            raise ValueError(
                "Project ID must be provided or set via PROJECT_ID environment variable."
            )
        
        self.region = region or os.environ.get("REGION")
        if not self.region:
            raise ValueError(
                "Region must be provided or set via REGION environment variable."
            )
        
        if model is None:
            raise ValueError("The 'model' argument is required and cannot be None.")
        
        # Set credentials environment variable
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        
        # Initialize Vertex AI
        vertexai.init(project=self.project_id, location=self.region)
        
        # Initialize the model
        self.model = GenerativeModel(model)
        self.model_name = model

    def generate(self, prompt: str, image_path: Union[str, Path]) -> str:
        """
        Generate content using the Vertex AI model with text + image as input.

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
            # Load the image using Vertex AI's Image class
            image = Image.load_from_file(str(image_path_obj))
            
            # Generate content using the model
            response = self.model.generate_content([prompt, image])
            return response.text

        except Exception as e:
            raise Exception(f"Failed to generate content with Vertex AI model: {e}")