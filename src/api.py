"""
FastAPI application for DermAId.
Provides endpoints:
  - GET  /health
  - POST /query  (multipart/form-data with optional image)
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from .logger import logger
from .exceptions import DermAidError, SessionNotFoundError, KnowledgeBaseError
from .agent import run_agent
from .config import settings

app = FastAPI(
    title="DermAId",
    description="Medical AI agent for dermatology",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/query")
async def query(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    strictness: Optional[float] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """
    Process a user query with optional image attachment.

    Returns JSON with `answer` and `session_id`.
    """
    # Use existing session or create new one
    sid = session_id or str(uuid.uuid4())
    logger.info("Query received: session=%s, query='%s...'", sid, query[:80])

    # Save uploaded image to a temporary directory if provided
    image_path = None
    if image:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        # Sanitize filename
        safe_name = f"{sid}_{uuid.uuid4().hex[:8]}_{image.filename}"
        image_path = str(upload_dir / safe_name)
        try:
            contents = await image.read()
            with open(image_path, "wb") as f:
                f.write(contents)
            logger.debug("Image saved to %s", image_path)
        except Exception as e:
            logger.exception("Failed to save uploaded image")
            raise HTTPException(status_code=500, detail="Could not save image.")

    # Execute agent
    try:
        result = run_agent(
            query=query,
            session_id=sid,
            strictness=strictness,
            image_path=image_path,
        )
    except DermAidError as e:
        logger.error("Agent failed: %s", e)
        raise HTTPException(status_code=500, detail=e.message)
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temporary image
        if image_path and Path(image_path).exists():
            Path(image_path).unlink(missing_ok=True)

    if result["error"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(content={
        "answer": result["answer"],
        "session_id": result["session_id"],
    })


def start():
    """Entry point for `dermaid` command."""
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()