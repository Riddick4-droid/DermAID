"""
Chroma vector‑store manager for the dermatology knowledge base.
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .logger import logger
from .exceptions import (
    KnowledgeBaseError,
    KnowledgeBaseEmptyError,
    KnowledgeBaseQueryError,
)
from .config import settings


class KnowledgeBase:
    """Manages a persistent Chroma collection with local embeddings."""

    def __init__(self, persist_directory: Optional[str] = None):
        # Allow override from config or default
        kb_config = settings.kb
        self.persist_directory = persist_directory or kb_config["persist_directory"]
        self.collection_name = kb_config["collection_name"]
        self.embedding_model = kb_config["embedding_model"]

        # Initialize Chroma client (persistent)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Embedding function: uses the same local model as our config
        self.embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )

        # Get or create the collection
        self.collection = self._get_or_create_collection()
        logger.info(
            "KnowledgeBase ready: collection='%s' (%d documents)",
            self.collection_name,
            self.collection.count(),
        )

    def _get_or_create_collection(self):
        """Return the existing collection or create one if it doesn't exist."""
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
            )
        except Exception:
            logger.info("Creating new Chroma collection '%s'", self.collection_name)
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
                metadata={"description": "Dermatology knowledge base"},
            )

    def add_chunks(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add a list of text chunks to the knowledge base.
        Each chunk can have associated metadata and a unique ID.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list")
            return

        # Default IDs if not provided
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]
        if metadatas is None:
            metadatas = [{}] * len(chunks)

        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info("Added %d chunks to KB (total now %d)", len(chunks), self.count())

    def query(self, query_text: str, k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve the top‑k most relevant chunks for a query.
        Returns a list of dicts with keys: 'text', 'metadata', 'score'.
        """
        if self.count() == 0:
            raise KnowledgeBaseEmptyError(self.collection_name)

        k = k or settings.retrieval["k"]
        try:
            results = self.collection.query(query_texts=[query_text], n_results=k)
        except Exception as exc:
            logger.exception("Query failed: %s", query_text)
            raise KnowledgeBaseQueryError(query_text, str(exc)) from exc

        # Transform Chroma result into a convenient list
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,  # Chroma returns distance; higher = more relevant
            })
        logger.debug("Query returned %d results for '%s...'", len(output), query_text[:50])
        return output
    
    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete the collection and recreate it empty."""
        self.client.delete_collection(self.collection_name)
        logger.warning("Deleted collection '%s'", self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Dermatology knowledge base"},
        )

def create_kb_from_text(text: str) -> KnowledgeBase:
    """
    Split a text into chunks using the configured separator,
    then index them into a fresh KnowledgeBase.
    Returns the populated KnowledgeBase instance.
    """
    kb = KnowledgeBase()
    kb.clear()  # start fresh

    separator = settings.kb["chunk_separator"]  # e.g., "\n## "
    raw_chunks = text.strip().split(separator)

    # Clean up first chunk if it lost the separator prefix
    if raw_chunks and not raw_chunks[0].startswith("## "):
        raw_chunks[0] = "## " + raw_chunks[0]

    chunks, metadatas, ids = [], [], []
    for chunk in raw_chunks:
        lines = chunk.strip().split("\n")
        if not lines:
            continue
        # Extract condition name from first line
        header = lines[0].replace("## ", "").strip().lower()
        condition = header.title()
        slug = header.replace(" ", "_")

        chunks.append(chunk.strip())
        metadatas.append({"condition": condition})
        ids.append(slug)

    kb.add_chunks(chunks, metadatas, ids)
    logger.info("Indexed %d chunks from text input.", len(chunks))
    return kb