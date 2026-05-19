import logging
from typing import Any, Dict

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import (
    BaseModel,
    Field,
)

from Services.chroma_db_service import (
    ChromaDBService,
    QueryExecutionError,
)

from Services.embedding_service import (
    EmbeddingService,
    EmbeddingGenerationError,
)


# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Router Configuration
# ---------------------------------------------------
retrieval_router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Vector Retrieval"],
)


# ---------------------------------------------------
# Singleton Services
# ---------------------------------------------------
# Avoid model/database reinitialization per request
embedding_service = EmbeddingService()

chroma_db_service = ChromaDBService(
    collection_name="test_collection"
)


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class QueryPayload(BaseModel):
    """
    Retrieval request payload.
    """

    query_text: str = Field(
        ...,
        min_length=2,
        max_length=5000,
        description="User search query",
        example="What is attention mechanism?",
    )

    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve",
    )


# ---------------------------------------------------
# Response Schema
# ---------------------------------------------------
class RetrievalResponse(BaseModel):
    """
    Standard retrieval API response.
    """

    success: bool
    query: str
    results: Dict[str, Any]


# ---------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------
@retrieval_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    logger.info(
        "Retrieval health check called."
    )

    return {
        "status": "healthy",
        "service": "Vector Retrieval API",
    }


# ---------------------------------------------------
# Retrieve Documents Endpoint
# ---------------------------------------------------
@retrieval_router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve(
    query_payload: QueryPayload,
) -> RetrievalResponse:
    """
    Retrieve relevant chunks from ChromaDB.

    Steps:
    - Generate query embedding
    - Perform vector similarity search
    - Return relevant chunks
    """

    try:
        logger.info(
            "Received retrieval request."
        )

        logger.info(
            "Query Text: %s",
            query_payload.query_text,
        )

        # ---------------------------------------------------
        # Step 1: Generate Embedding
        # ---------------------------------------------------
        logger.info(
            "Generating query embedding..."
        )

        embedding = (
            embedding_service.embed(
                query_payload.query_text
            )
        )

        logger.info(
            "Embedding generated successfully."
        )

        # ---------------------------------------------------
        # Step 2: Query ChromaDB
        # ---------------------------------------------------
        logger.info(
            "Performing vector similarity search..."
        )

        result = (
            chroma_db_service.query_documents(
                query_embeddings=embedding,
                n_results=query_payload.n_results,
            )
        )

        logger.info(
            "Retrieval completed successfully."
        )

        return RetrievalResponse(
            success=True,
            query=query_payload.query_text,
            results=result,
        )

    except EmbeddingGenerationError as embedding_error:
        logger.exception(
            "Embedding generation failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "success": False,
                "message": str(
                    embedding_error
                ),
            },
        )

    except QueryExecutionError as retrieval_error:
        logger.exception(
            "Vector retrieval failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "success": False,
                "message": str(
                    retrieval_error
                ),
            },
        )

    except Exception as e:
        logger.exception(
            "Unexpected server error occurred."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "success": False,
                "message": (
                    "Internal server error."
                ),
            },
        ) from e