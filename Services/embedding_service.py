import logging
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


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
class EmbeddingServiceError(Exception):
    """Base exception for embedding service."""


class ModelInitializationError(EmbeddingServiceError):
    """Raised when embedding model fails to load."""


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embeddings cannot be generated."""


# ---------------------------------------------------
# Embedding Service
# ---------------------------------------------------
class EmbeddingService:
    """
    Production-grade embedding service using Sentence Transformers.

    Features:
    - Singleton model loading
    - Logging
    - Exception handling
    - Input validation
    - Batch embedding support
    - Configurable model path
    """

    def __init__(
        self,
        model_path: str = "Embedding_Model/paraphrase-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize embedding model.

        Args:
            model_path (str): Path or HuggingFace model name
        """

        try:
            logger.info(
                "Loading embedding model: %s",
                model_path,
            )

            self.model = SentenceTransformer(model_path)

            logger.info(
                "Embedding model loaded successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to initialize embedding model."
            )

            raise ModelInitializationError(
                f"Model loading failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Generate Embeddings
    # ---------------------------------------------------
    def embed(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for input text.

        Args:
            sentences:
                Single sentence or list of sentences

            batch_size:
                Batch size for inference

            normalize_embeddings:
                Whether to normalize vectors

        Returns:
            List[List[float]]: Vector embeddings
        """

        try:
            logger.info("Validating embedding input...")

            # ---------------- Validation ----------------
            if not sentences:
                raise ValueError(
                    "Input sentences cannot be empty."
                )

            # Convert single string to list
            if isinstance(sentences, str):
                sentences = [sentences]

            if not isinstance(sentences, list):
                raise TypeError(
                    "Sentences must be a string or list of strings."
                )

            # Remove empty sentences
            sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence and sentence.strip()
            ]

            if not sentences:
                raise ValueError(
                    "No valid sentences found after cleaning."
                )

            logger.info(
                "Generating embeddings for %d sentence(s)...",
                len(sentences),
            )

            embeddings = self.model.encode(
                sentences,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            logger.info(
                "Embedding generation completed successfully."
            )

            return embeddings.tolist()

        except ValueError as ve:
            logger.warning(
                "Validation error during embedding: %s",
                str(ve),
            )

            raise EmbeddingGenerationError(
                str(ve)
            ) from ve

        except Exception as e:
            logger.exception(
                "Unexpected error during embedding generation."
            )

            raise EmbeddingGenerationError(
                f"Embedding generation failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Get Embedding Dimension
    # ---------------------------------------------------
    def get_embedding_dimension(self) -> int:
        """
        Get embedding vector dimension.

        Returns:
            int: Embedding dimension
        """

        try:
            dimension = self.model.get_sentence_embedding_dimension()

            logger.info(
                "Embedding dimension fetched successfully: %d",
                dimension,
            )

            return dimension

        except Exception as e:
            logger.exception(
                "Failed to fetch embedding dimension."
            )

            raise EmbeddingServiceError(
                f"Could not fetch embedding dimension: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":

    try:
        embedding_service = EmbeddingService()

        sample_sentences = [
            "Sonakshi is a good girl.",
            "Why is she not a bad girl?",
        ]

        embeddings = embedding_service.embed(
            sample_sentences
        )

        print("\nGenerated Embeddings:")
        print("-" * 100)

        for index, embedding in enumerate(
            embeddings,
            start=1,
        ):
            print(f"\nSentence {index}")
            print(f"Vector Length: {len(embedding)}")
            print(embedding[:10])  # Preview first 10 values

        print("\nEmbedding Dimension:")
        print(
            embedding_service.get_embedding_dimension()
        )

    except EmbeddingServiceError as e:
        logger.error(
            "Application failed: %s",
            str(e),
        )