import logging
from pathlib import Path
from typing import List

from Services.document_handler import DocumentHandler
from Services.embedding_service import EmbeddingService
from Services.chroma_db_service import ChromaDBService


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
class IngestionServiceError(Exception):
    """Base exception for ingestion service."""


class DocumentIngestionError(IngestionServiceError):
    """Raised when document ingestion fails."""


# ---------------------------------------------------
# Ingestion Service
# ---------------------------------------------------
class IngestionService:
    """
    Production-grade document ingestion pipeline.

    Pipeline Flow:
    1. Read document
    2. Clean text
    3. Generate chunks
    4. Create embeddings
    5. Store in ChromaDB

    Features:
    - Logging
    - Exception handling
    - Validation
    - Modular architecture
    - Clean service orchestration
    """

    def __init__(self, collection_name: str):
        """
        Initialize ingestion pipeline services.

        Args:
            collection_name (str): ChromaDB collection name
        """

        try:
            logger.info(
                "Initializing ingestion service..."
            )

            self.document_handler = DocumentHandler()

            self.embedding_service = EmbeddingService()

            self.chroma_db_service = ChromaDBService(
                collection_name
            )

            logger.info(
                "Ingestion service initialized successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to initialize ingestion service."
            )

            raise IngestionServiceError(
                f"Initialization failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Ingest Document
    # ---------------------------------------------------
    def ingest_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> bool:
        """
        Ingest document into vector database.

        Steps:
        - Read PDF
        - Generate chunks
        - Create embeddings
        - Store vectors in ChromaDB

        Args:
            file_path (str):
                Path to document

            chunk_size (int):
                Chunk size

            chunk_overlap (int):
                Overlap between chunks

        Returns:
            bool:
                True if ingestion successful
        """

        try:
            logger.info(
                "Starting ingestion for file: %s",
                file_path,
            )

            # ---------------- Validation ----------------
            path = Path(file_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"File does not exist: {file_path}"
                )

            if chunk_size <= 0:
                raise ValueError(
                    "chunk_size must be greater than 0."
                )

            if chunk_overlap < 0:
                raise ValueError(
                    "chunk_overlap cannot be negative."
                )

            if chunk_overlap >= chunk_size:
                raise ValueError(
                    "chunk_overlap must be smaller than chunk_size."
                )

            # ---------------------------------------------------
            # Step 1: Generate Chunks
            # ---------------------------------------------------
            logger.info(
                "Generating text chunks..."
            )

            chunks = self.document_handler.get_chunks(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                raise ValueError(
                    "No chunks generated from document."
                )

            logger.info(
                "Generated %d chunks successfully.",
                len(chunks),
            )

            # ---------------------------------------------------
            # Step 2: Generate IDs
            # ---------------------------------------------------
            logger.info(
                "Generating document IDs and metadata..."
            )

            ids = [
                f"{path.stem}_chunk_{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {
                    "file_name": path.name,
                    "file_path": str(path),
                    "chunk_index": i,
                }
                for i in range(len(chunks))
            ]

            # ---------------------------------------------------
            # Step 3: Generate Embeddings
            # ---------------------------------------------------
            logger.info(
                "Generating embeddings..."
            )

            embeddings = self.embedding_service.embed(
                chunks
            )

            logger.info(
                "Generated embeddings successfully."
            )

            # ---------------------------------------------------
            # Step 4: Store in ChromaDB
            # ---------------------------------------------------
            logger.info(
                "Storing embeddings in ChromaDB..."
            )

            self.chroma_db_service.add_document(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            logger.info(
                "Document ingestion completed successfully."
            )

            return True

        except FileNotFoundError as fnf:
            logger.exception(
                "Document file not found."
            )

            raise DocumentIngestionError(
                str(fnf)
            ) from fnf

        except ValueError as ve:
            logger.exception(
                "Validation error during ingestion."
            )

            raise DocumentIngestionError(
                str(ve)
            ) from ve

        except Exception as e:
            logger.exception(
                "Unexpected error during document ingestion."
            )

            raise DocumentIngestionError(
                f"Ingestion failed: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":

    try:
        ingestion_service = IngestionService(
            collection_name="rag_documents"
        )

        ingestion_service.ingest_document(
            file_path="Documents/AttentionIsAllYouNeed.pdf",
            chunk_size=500,
            chunk_overlap=100,
        )

        logger.info(
            "Document ingestion pipeline completed."
        )

    except IngestionServiceError as e:
        logger.error(
            "Application failed: %s",
            str(e),
        )