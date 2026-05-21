import logging
from typing import Any, Dict
from Guardrails.input_guardrails import InputGuardrails

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import (
    BaseModel,
    Field,
)

from Utils.get_answer import (
    RAGPipelineError,
    RAGService,
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
# FastAPI Router
# ---------------------------------------------------
get_answer_router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG QA System"],
)


# ---------------------------------------------------
# Singleton RAG Service
# ---------------------------------------------------
# Avoid re-initializing models/services per request
rag_service = RAGService(collection_name="test_collection")
input_guardrail = InputGuardrails()


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class QueryPayload(BaseModel):
    """
    Request payload schema.
    """

    query: str = Field(
        ...,
        min_length=2,
        max_length=5000,
        description="User question/query",
        example="What is attention mechanism?",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve",
    )


# ---------------------------------------------------
# Response Schema
# ---------------------------------------------------
class QueryResponse(BaseModel):
    """
    API response schema.
    """

    success: bool
    query: str
    answer: str


# ---------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------
@get_answer_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    logger.info("Health check endpoint called.")

    return {
        "status": "healthy",
        "service": "RAG API",
    }


# ---------------------------------------------------
# Get Answer Endpoint
# ---------------------------------------------------
@get_answer_router.post(
    "/get_answer",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
async def get_answer_route(
    query_payload: QueryPayload,
) -> QueryResponse:
    """
    Generate answer using RAG pipeline.

    Steps:
    - Retrieve relevant chunks
    - Augment prompt
    - Generate final answer
    """

    try:
        logger.info("Received query request.")

        logger.info(
            "User Query: %s",
            query_payload.query,
        )

        input_query = query_payload.query
        clean_query, guardrail_triggered = input_guardrail.run_input_guardrails(
            input_query
        )

        answer = clean_query
        if not guardrail_triggered:

            # ---------------------------------------------------
            # Generate Answer
            # ---------------------------------------------------
            answer = await rag_service.get_answer(
                query=clean_query,
                top_k=query_payload.top_k,
            )

        logger.info("Answer generated successfully.")

        return QueryResponse(
            success=True,
            query=query_payload.query,
            answer=answer,
        )

    except RAGPipelineError as rag_error:
        logger.exception("RAG pipeline error occurred.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": str(rag_error),
            },
        )

    except Exception as e:
        logger.exception("Unexpected server error occurred.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "message": ("Internal server error."),
            },
        ) from e
