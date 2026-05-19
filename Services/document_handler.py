import logging
import re
from pathlib import Path
from typing import List

import pypdf
from pypdf.errors import PdfReadError


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
class DocumentHandlerError(Exception):
    """Base exception for document handler."""


class DocumentReadError(DocumentHandlerError):
    """Raised when PDF reading fails."""


class TextCleaningError(DocumentHandlerError):
    """Raised when text cleaning fails."""


class ChunkGenerationError(DocumentHandlerError):
    """Raised when chunk generation fails."""


# ---------------------------------------------------
# Document Handler
# ---------------------------------------------------
class DocumentHandler:
    """
    Production-grade PDF document handler.

    Features:
    - PDF reading
    - Text cleaning
    - Chunk generation
    - Exception handling
    - Logging
    - Validation
    """

    # ---------------------------------------------------
    # Read PDF Document
    # ---------------------------------------------------
    def __read_document(self, file_path: str) -> str:
        """
        Read PDF document and extract text.

        Args:
            file_path (str): Path to PDF file

        Returns:
            str: Extracted text
        """

        try:
            logger.info("Reading PDF document: %s", file_path)

            pdf_path = Path(file_path)

            # ---------------- Validation ----------------
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"File does not exist: {file_path}"
                )

            if pdf_path.suffix.lower() != ".pdf":
                raise ValueError(
                    "Only PDF files are supported."
                )

            reader = pypdf.PdfReader(file_path)

            if not reader.pages:
                raise ValueError(
                    "PDF contains no readable pages."
                )

            extracted_text = []

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    page_text = page.extract_text()

                    if page_text:
                        extracted_text.append(page_text)

                    logger.info(
                        "Processed page %d successfully.",
                        page_number,
                    )

                except Exception as page_error:
                    logger.warning(
                        "Failed to process page %d: %s",
                        page_number,
                        str(page_error),
                    )

            final_text = " ".join(extracted_text)

            if not final_text.strip():
                raise ValueError(
                    "No extractable text found in PDF."
                )

            logger.info(
                "PDF text extraction completed successfully."
            )

            return final_text

        except FileNotFoundError as fnf:
            logger.exception("PDF file not found.")
            raise DocumentReadError(str(fnf)) from fnf

        except PdfReadError as pdf_error:
            logger.exception("Invalid or corrupted PDF.")
            raise DocumentReadError(
                f"PDF reading failed: {str(pdf_error)}"
            ) from pdf_error

        except Exception as e:
            logger.exception("Unexpected error during PDF reading.")
            raise DocumentReadError(
                f"Unexpected document read error: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Clean Text
    # ---------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Args:
            text (str): Raw extracted text

        Returns:
            str: Cleaned text
        """

        try:
            logger.info("Cleaning extracted text...")

            if not text:
                raise ValueError("Input text cannot be empty.")

            # Replace multiple spaces/newlines/tabs
            text = re.sub(r"\s+", " ", text)

            # Remove unwanted special characters if needed
            text = text.strip()

            logger.info("Text cleaning completed.")

            return text

        except Exception as e:
            logger.exception("Text cleaning failed.")
            raise TextCleaningError(
                f"Text cleaning error: {str(e)}"
            ) from e

    # ---------------------------------------------------
    # Generate Chunks
    # ---------------------------------------------------
    def get_chunks(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> List[str]:
        """
        Split document into chunks.

        Args:
            file_path (str): PDF file path
            chunk_size (int): Size of each chunk
            chunk_overlap (int): Overlap between chunks

        Returns:
            List[str]: List of text chunks
        """

        try:
            logger.info("Starting chunk generation...")

            # ---------------- Validation ----------------
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

            # Read + clean text
            text = self.__read_document(file_path)
            text = self.clean_text(text)

            chunks = []

            step = chunk_size - chunk_overlap

            for i in range(0, len(text), step):
                chunk = text[i : i + chunk_size]

                if chunk.strip():
                    chunks.append(chunk)

            logger.info(
                "Chunk generation completed successfully. Total chunks: %d",
                len(chunks),
            )

            return chunks

        except Exception as e:
            logger.exception("Chunk generation failed.")
            raise ChunkGenerationError(
                f"Chunk generation error: {str(e)}"
            ) from e


# ---------------------------------------------------
# Example Usage
# ---------------------------------------------------
if __name__ == "__main__":

    try:
        document_handler = DocumentHandler()

        chunks = document_handler.get_chunks(
            file_path="Documents/AttentionIsAllYouNeed.pdf",
            chunk_size=500,
            chunk_overlap=100,
        )

        logger.info("Printing generated chunks...")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}")
            print("-" * 100)
            print(chunk)

    except DocumentHandlerError as e:
        logger.error("Application failed: %s", str(e))