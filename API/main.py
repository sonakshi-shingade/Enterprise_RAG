import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ---------------------------------------------------
# Environment Variables
# ---------------------------------------------------
load_dotenv()


# ---------------------------------------------------
# Project Root Path Setup
# ---------------------------------------------------
# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from frontend.TelegramBot import telegram_bot

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Import Routers
# ---------------------------------------------------
from Routes.ingest_route import (
    ingest_router,
)

from Routes.retriever_route import (
    retrieval_router,
)

from Routes.get_answer_route import (
    get_answer_router,
)


# ---------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown events.
    """

    try:
        logger.info("Starting RAG Application...")

        # Initialize Telegram bot in the background
        # ---------------- Startup ----------------
        app.state.telegram_bot_task = await telegram_bot.start_telegram_bot()
        if app.state.telegram_bot_task is None:
            logger.warning("Telegram bot was not started")

        logger.info("Application startup completed.")

        logger.info("Starting RAG Application...")

        logger.info("Application startup completed.")

        yield

        # ---------------- Shutdown ----------------
        logger.info("Shutting down RAG Application...")

        logger.info("Application shutdown completed.")

    except Exception as e:
        logger.exception(f"Error during application startup: {e}")
        raise e

    finally:
        if getattr(app.state, "telegram_bot_task", None) is not None:
            await telegram_bot.stop_telegram_bot(app.state.telegram_bot_task)

        logger.info("Shutting down RAG Application...")


# ---------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------
app = FastAPI(
    title="Enterprise RAG API",
    description=(
        "Production-grade Retrieval-Augmented "
        "Generation (RAG) API using FastAPI, "
        "ChromaDB, Sentence Transformers, "
        "and OpenRouter."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------
# CORS Middleware
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------
# Register Routers
# ---------------------------------------------------
app.include_router(ingest_router)

app.include_router(retrieval_router)

app.include_router(get_answer_router)


# ---------------------------------------------------
# Root Endpoint
# ---------------------------------------------------
@app.get("/")
async def root():
    """
    Root endpoint.
    """

    logger.info("Root endpoint called.")

    return {
        "message": ("Enterprise RAG API is running successfully."),
        "version": "1.0.0",
        "status": "healthy",
    }


# ---------------------------------------------------
# Global Health Check
# ---------------------------------------------------
@app.get("/health")
async def health_check():
    """
    Global application health check.
    """

    logger.info("Global health check called.")

    return {
        "status": "healthy",
        "application": "Enterprise RAG API",
    }


# ---------------------------------------------------
# Run Application
# ---------------------------------------------------
if __name__ == "__main__":

    logger.info("Starting Uvicorn server...")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8200,
        reload=False,  # Disable in production
        workers=1,  # Increase in production
        log_level="info",
    )
