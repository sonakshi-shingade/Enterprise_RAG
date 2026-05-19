import logging
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.errors import ChromaError


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
class ChromaDBServiceError(Exception):
    """Base exception for ChromaDB service."""


class DocumentInsertionError(ChromaDBServiceError):
    """Raised when document insertion fails."""


class QueryExecutionError(ChromaDBServiceError):
    """Raised when query execution fails."""


# ---------------------------------------------------
# ChromaDB Service
# ---------------------------------------------------
class ChromaDBService:
    """
    Production-grade ChromaDB Service Layer.
    Handles:
    - Persistent ChromaDB connection
    - Collection management
    - Logging
    - Exception handling
    - Input validation
    """

    def __init__(
        self,
        collection_name: str,
        db_path: str = "./Database",
    ) -> None:
        """
        Initialize ChromaDB client and collection.

        Args:
            collection_name (str): Name of the collection
            db_path (str): Path for persistent storage
        """

        try:
            logger.info("Initializing ChromaDB client...")

            self.client = chromadb.PersistentClient(path=db_path)

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "Collection '%s' initialized successfully.",
                collection_name,
            )

        except ChromaError as ce:
            logger.exception("ChromaDB initialization failed.")
            raise ChromaDBServiceError(
                f"Failed to initialize ChromaDB: {str(ce)}"
            ) from ce

        except Exception as e:
            logger.exception("Unexpected error during initialization.")
            raise ChromaDBServiceError(
                f"Unexpected initialization error: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Add Documents
    # ---------------------------------------------------
    def add_document(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Add documents to ChromaDB collection.

        Args:
            ids: Unique document IDs
            embeddings: Vector embeddings
            documents: Document text
            metadatas: Optional metadata

        Returns:
            bool: True if insertion succeeds
        """

        try:
            logger.info("Validating input before insertion...")

            # ---------------- Validation ----------------
            if not ids:
                raise ValueError("IDs list cannot be empty.")

            if not embeddings:
                raise ValueError("Embeddings list cannot be empty.")

            if not documents:
                raise ValueError("Documents list cannot be empty.")

            if not (
                len(ids)
                == len(embeddings)
                == len(documents)
            ):
                raise ValueError(
                    "Mismatch in lengths of ids, embeddings, and documents."
                )

            if metadatas and len(metadatas) != len(ids):
                raise ValueError(
                    "Metadata count must match IDs count."
                )

            logger.info(
                "Adding %d documents to collection...",
                len(ids),
            )

            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "%d documents added successfully.",
                len(ids),
            )

            return True

        except ValueError as ve:
            logger.warning("Validation error: %s", str(ve))
            raise DocumentInsertionError(str(ve)) from ve

        except ChromaError as ce:
            logger.exception("ChromaDB insertion failed.")
            raise DocumentInsertionError(
                f"Failed to insert documents: {str(ce)}"
            ) from ce

        except Exception as e:
            logger.exception("Unexpected insertion error.")
            raise DocumentInsertionError(
                f"Unexpected insertion error: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Query Documents
    # ---------------------------------------------------
    def query_documents(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query similar documents from ChromaDB.

        Args:
            query_embeddings: Query vector embeddings
            n_results: Number of results to fetch

        Returns:
            Dict[str, Any]: Query response
        """

        try:
            logger.info("Validating query input...")

            # ---------------- Validation ----------------
            if not query_embeddings:
                raise ValueError(
                    "Query embeddings cannot be empty."
                )

            if n_results <= 0:
                raise ValueError(
                    "n_results must be greater than 0."
                )

            logger.info(
                "Executing similarity search with top_k=%d",
                n_results,
            )

            result = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
            )

            logger.info("Query executed successfully.")

            return result

        except ValueError as ve:
            logger.warning("Validation error: %s", str(ve))
            raise QueryExecutionError(str(ve)) from ve

        except ChromaError as ce:
            logger.exception("ChromaDB query failed.")
            raise QueryExecutionError(
                f"Failed to query documents: {str(ce)}"
            ) from ce

        except Exception as e:
            logger.exception("Unexpected query error.")
            raise QueryExecutionError(
                f"Unexpected query error: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":
    try:
        chroma_service = ChromaDBService(
            collection_name="documents_collection"
        )

        chroma_service.add_document(
            ids=["1"],
            embeddings=[[0.1, 0.2, 0.3]],
            documents=["This is a sample document"],
            metadatas=[{"source": "test"}],
        )

        result = chroma_service.query_documents(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2,
        )

        print(result)

    except ChromaDBServiceError as e:
        logger.error("Application failed: %s", str(e))