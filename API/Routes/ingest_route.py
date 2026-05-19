import logging
from typing import Dict

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from Services.ingestion_service import (
    IngestionService,
    IngestionServiceError,
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
ingest_router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["Document Ingestion"],
)


# ---------------------------------------------------
# Singleton Service
# ---------------------------------------------------
# Prevent reloading services/models repeatedly
ingestion_service = IngestionService(
    collection_name="test_collection"
)


# ---------------------------------------------------
# Request Schema
# ---------------------------------------------------
class IngestPayload(BaseModel):
    """
    Request payload for document ingestion.
    """

    file_path: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Path to the PDF document",
        example="Documents/AttentionIsAllYouNeed.pdf",
    )

    chunk_size: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Chunk size for splitting document",
    )

    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Overlap between chunks",
    )

    # ---------------------------------------------------
    # Custom Validation
    # ---------------------------------------------------
    @field_validator("file_path")
    @classmethod
    def validate_pdf_file(
        cls,
        value: str,
    ) -> str:

        if not value.lower().endswith(".pdf"):
            raise ValueError(
                "Only PDF files are supported."
            )

        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(
        cls,
        value: int,
        info,
    ) -> int:

        chunk_size = info.data.get(
            "chunk_size",
            500,
        )

        if value >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        return value


# ---------------------------------------------------
# Response Schema
# ---------------------------------------------------
class IngestResponse(BaseModel):
    """
    Standard API response schema.
    """

    success: bool
    message: str
    file_path: str


# ---------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------
@ingest_router.get(
    "/health",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """

    logger.info(
        "Ingestion health check called."
    )

    return {
        "status": "healthy",
        "service": "Document Ingestion API",
    }


# ---------------------------------------------------
# Ingest Document Endpoint
# ---------------------------------------------------
@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_document(
    ingest_payload: IngestPayload,
) -> IngestResponse:
    """
    Ingest PDF document into vector database.

    Steps:
    - Read PDF
    - Generate chunks
    - Create embeddings
    - Store in ChromaDB
    """

    try:
        logger.info(
            "Received ingestion request."
        )

        logger.info(
            "File Path: %s",
            ingest_payload.file_path,
        )

        # ---------------------------------------------------
        # Start Ingestion
        # ---------------------------------------------------
        ingestion_service.ingest_document(
            file_path=ingest_payload.file_path,
            chunk_size=ingest_payload.chunk_size,
            chunk_overlap=ingest_payload.chunk_overlap,
        )

        logger.info(
            "Document ingested successfully."
        )

        return IngestResponse(
            success=True,
            message=(
                "Document ingested successfully."
            ),
            file_path=(
                ingest_payload.file_path
            ),
        )

    except IngestionServiceError as ingestion_error:
        logger.exception(
            "Document ingestion failed."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail={
                "success": False,
                "message": str(
                    ingestion_error
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