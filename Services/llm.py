import json
import logging
import os
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()


# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------
class LLMServiceError(Exception):
    """Base exception for LLM service."""


class APIKeyMissingError(LLMServiceError):
    """Raised when API key is missing."""


class APIRequestError(LLMServiceError):
    """Raised when API request fails."""


class InvalidResponseError(LLMServiceError):
    """Raised when invalid response is received."""


# ---------------------------------------------------
# LLM Service
# ---------------------------------------------------
class LLMService:
    """
    Production-grade OpenRouter LLM Service.

    Features:
    - Secure API key handling
    - Request timeout
    - Structured logging
    - Exception handling
    - Configurable models
    - Temperature support
    - JSON-safe responses
    - HTTP session reuse
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self) -> None:
        """
        Initialize LLM service.
        """

        try:
            logger.info(
                "Initializing LLM service..."
            )

            self.api_key = os.getenv(
                "OPEN_ROUTER_KEY"
            )

            if not self.api_key:
                raise APIKeyMissingError(
                    "OPEN_ROUTER_KEY not found in environment variables."
                )

            # Reusable HTTP session
            self.session = requests.Session()

            logger.info(
                "LLM service initialized successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to initialize LLM service."
            )

            raise LLMServiceError(
                f"Initialization failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Generate Response
    # ---------------------------------------------------
    def get_response(
        self,
        system_prompt: str,
        user_input: str,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> str:
        """
        Generate LLM response.

        Args:
            system_prompt (str):
                System instructions

            user_input (str):
                User query

            model (str):
                LLM model name

            temperature (float):
                Creativity/randomness

            max_tokens (int):
                Maximum output tokens

            timeout (int):
                API timeout in seconds

        Returns:
            str:
                Model response
        """

        try:
            logger.info(
                "Validating LLM request..."
            )

            # ---------------- Validation ----------------
            if not system_prompt.strip():
                raise ValueError(
                    "System prompt cannot be empty."
                )

            if not user_input.strip():
                raise ValueError(
                    "User input cannot be empty."
                )

            logger.info(
                "Sending request to model: %s",
                model,
            )

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = self.session.post(
                url=self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Raise HTTP errors
            response.raise_for_status()

            logger.info(
                "Response received successfully from OpenRouter."
            )

            response_data = response.json()

            # ---------------- Response Validation ----------------
            if (
                "choices" not in response_data
                or not response_data["choices"]
            ):
                raise InvalidResponseError(
                    "Invalid response structure received from API."
                )

            content = (
                response_data["choices"][0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                raise InvalidResponseError(
                    "Empty response received from model."
                )

            logger.info(
                "LLM response generated successfully."
            )

            return content

        except ValueError as ve:
            logger.warning(
                "Validation error: %s",
                str(ve),
            )

            raise APIRequestError(
                str(ve)
            ) from ve

        except Timeout as timeout_error:
            logger.exception(
                "LLM request timed out."
            )

            raise APIRequestError(
                f"Request timeout: {str(timeout_error)}"
            ) from timeout_error

        except ConnectionError as connection_error:
            logger.exception(
                "Connection error occurred."
            )

            raise APIRequestError(
                f"Connection error: {str(connection_error)}"
            ) from connection_error

        except HTTPError as http_error:
            logger.exception(
                "HTTP error occurred."
            )

            raise APIRequestError(
                f"HTTP error: {str(http_error)}"
            ) from http_error

        except RequestException as request_error:
            logger.exception(
                "General request exception occurred."
            )

            raise APIRequestError(
                f"API request failed: {str(request_error)}"
            ) from request_error

        except json.JSONDecodeError as json_error:
            logger.exception(
                "Invalid JSON response."
            )

            raise InvalidResponseError(
                f"Invalid JSON response: {str(json_error)}"
            ) from json_error

        except Exception as e:
            logger.exception(
                "Unexpected error during LLM response generation."
            )

            raise LLMServiceError(
                f"Unexpected LLM service error: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":

    try:
        llm_service = LLMService()

        system_prompt = (
            "You are a helpful AI assistant."
        )

        user_input = (
            "Answer in JSON only: "
            "What is the capital of France?"
        )

        response = llm_service.get_response(
            system_prompt=system_prompt,
            user_input=user_input,
            temperature=0.2,
            max_tokens=200,
        )

        print("\nGenerated Response:")
        print("-" * 100)
        print(response)

    except LLMServiceError as e:
        logger.error(
            "Application failed: %s",
            str(e),
        )