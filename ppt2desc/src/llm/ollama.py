"""
@file ollama.py
@brief Ollama local LLM client implementation
@details This file contains a Python class for interacting with an Ollama instance running on localhost.
@version 1.0
@date 9 Jan 2025
@author You
"""

import requests
import json
from pathlib import Path
from typing import Union

from .base import LLMClient

# @class OllamaClient
# @brief A client for locally hosted LLM inference via Ollama.
# @details This client calls the Ollama REST API at http://localhost:11434 by default.
class OllamaClient(LLMClient):
    # @brief Constructor for OllamaClient
    # @param model The local model name to run in Ollama (e.g. "llava:34b-v1.6-fp16")
    # @param host The base URL for the Ollama service (default "http://localhost:11434")
    # @param kwargs Additional future parameters (not used here)
    def __init__(self, model: str = "llava:34b-v1.6-fp16", host: str = "http://localhost:11434", **kwargs):
        self.model_name = model
        self.host = host

    # @brief Generate text from a given prompt (and optionally an image, if your pipeline requires it).
    # @param prompt Text prompt to send to Ollama
    # @param image_path (optional) If your pipeline calls generate() with an image, this argument is included. We ignore it here, unless you adapt Ollama to do image-based tasks.
    # @return The generated text from the local Ollama model
    def generate(self, prompt: str, image_path: Union[str, Path, None] = None) -> str:
        """
        Construct request payload and POST to the Ollama server at self.host/generate.
        Stream the JSON lines and combine them into a final string.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "num_ctx": 2048,    # Typical context window
            "max_tokens": 512   # Adjust as needed
        }

        try:
            response = requests.post(
                f"{self.host}/generate",
                json=payload,
                stream=True,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            final_text = []
            # Ollama streams line-by-line in JSON
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if "done" in data:
                            break
                        if "response" in data:
                            final_text.append(data["response"])
                    except json.JSONDecodeError:
                        # Ignore non-JSON lines
                        pass

            return "".join(final_text)

        except requests.RequestException as e:
            return f"[ERROR: Ollama request failed] {str(e)}"

