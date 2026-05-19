import logging
from typing import List

from Services.chroma_db_service import ChromaDBService
from Services.embedding_service import (
    EmbeddingService,
)
from Services.llm import LLMService


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
class RAGPipelineError(Exception):
    """Base exception for RAG pipeline."""


class RetrievalError(RAGPipelineError):
    """Raised when retrieval fails."""


class AugmentationError(RAGPipelineError):
    """Raised when augmentation fails."""


class GenerationError(RAGPipelineError):
    """Raised when generation fails."""


# ---------------------------------------------------
# RAG Service
# ---------------------------------------------------
class RAGService:
    """
    Production-grade RAG Pipeline.

    Pipeline:
    1. Query Embedding
    2. Vector Retrieval
    3. Context Augmentation
    4. LLM Generation

    Features:
    - Logging
    - Exception handling
    - Clean architecture
    - Async-ready structure
    - Modular services
    """

    def __init__(
        self,
        collection_name: str = "test_collection",
    ) -> None:
        """
        Initialize RAG services.
        """

        try:
            logger.info(
                "Initializing RAG pipeline..."
            )

            self.embedding_service = (
                EmbeddingService()
            )

            self.chroma_service = (
                ChromaDBService(
                    collection_name
                )
            )

            self.llm_service = LLMService()

            logger.info(
                "RAG pipeline initialized successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to initialize RAG pipeline."
            )

            raise RAGPipelineError(
                f"Initialization failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Retrieve Relevant Chunks
    # ---------------------------------------------------
    async def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """
        Retrieve relevant chunks from vector DB.

        Args:
            query (str):
                User query

            top_k (int):
                Number of documents to retrieve

        Returns:
            List[str]:
                Relevant document chunks
        """

        try:
            logger.info(
                "Starting retrieval phase..."
            )

            if not query.strip():
                raise ValueError(
                    "Query cannot be empty."
                )

            # Generate query embedding
            logger.info(
                "Generating query embedding..."
            )

            embedding = (
                self.embedding_service.embed(
                    query
                )
            )

            # Query vector DB
            logger.info(
                "Querying ChromaDB..."
            )

            results = (
                self.chroma_service.query_documents(
                    query_embeddings=embedding,
                    n_results=top_k,
                )
            )

            documents = results.get(
                "documents",
                []
            )

            if (
                not documents
                or not documents[0]
            ):
                raise RetrievalError(
                    "No relevant documents found."
                )

            logger.info(
                "Retrieved %d relevant chunks.",
                len(documents[0]),
            )

            return documents[0]

        except Exception as e:
            logger.exception(
                "Retrieval phase failed."
            )

            raise RetrievalError(
                f"Retrieval failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Build Prompt
    # ---------------------------------------------------
    def build_prompt(
        self,
        relevant_chunks: List[str],
    ) -> str:
        """
        Build RAG prompt using retrieved context.

        Args:
            relevant_chunks (List[str]):
                Retrieved chunks

        Returns:
            str:
                Final system prompt
        """

        try:
            logger.info(
                "Building augmented prompt..."
            )

            if not relevant_chunks:
                raise ValueError(
                    "Relevant chunks cannot be empty."
                )

            context = "\n\n".join(
                relevant_chunks
            )

            prompt = f"""
You are an intelligent AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not present in the context,
say:
"I could not find the answer in the provided context."

---------------- CONTEXT ----------------
{context}
------------------------------------------------
"""

            logger.info(
                "Prompt built successfully."
            )

            return prompt.strip()

        except Exception as e:
            logger.exception(
                "Prompt augmentation failed."
            )

            raise AugmentationError(
                f"Prompt building failed: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Generate Final Answer
    # ---------------------------------------------------
    async def get_answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Execute complete RAG pipeline.

        Args:
            query (str):
                User question

            top_k (int):
                Number of retrieved chunks

        Returns:
            str:
                Generated answer
        """

        try:
            logger.info(
                "Starting RAG pipeline..."
            )

            # ---------------------------------------------------
            # Step 1: Retrieval
            # ---------------------------------------------------
            relevant_chunks = (
                await self.retrieve_context(
                    query=query,
                    top_k=top_k,
                )
            )

            # ---------------------------------------------------
            # Step 2: Augmentation
            # ---------------------------------------------------
            system_prompt = (
                self.build_prompt(
                    relevant_chunks
                )
            )

            # ---------------------------------------------------
            # Step 3: Generation
            # ---------------------------------------------------
            logger.info(
                "Generating final response from LLM..."
            )

            response = (
                self.llm_service.get_response(
                    system_prompt=system_prompt,
                    user_input=query,
                )
            )

            logger.info(
                "RAG pipeline completed successfully."
            )

            return response

        except RetrievalError:
            raise

        except AugmentationError:
            raise

        except Exception as e:
            logger.exception(
                "Generation phase failed."
            )

            raise GenerationError(
                f"Answer generation failed: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":

    import asyncio

    async def main():

        try:
            rag_service = RAGService(
                collection_name="test_collection"
            )

            query = (
                "What is attention mechanism?"
            )

            answer = (
                await rag_service.get_answer(
                    query=query,
                    top_k=5,
                )
            )

            print("\nGenerated Answer:")
            print("-" * 100)
            print(answer)

        except RAGPipelineError as e:
            logger.error(
                "Application failed: %s",
                str(e),
            )

    asyncio.run(main())