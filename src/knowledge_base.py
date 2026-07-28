"""
Chroma vector‑store manager + PDF ingestion with configurable parsing.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .logger import logger
from .exceptions import (
    KnowledgeBaseError,
    KnowledgeBaseEmptyError,
    KnowledgeBaseQueryError,
)
from .config import settings


class KnowledgeBase:
    """Manages persistent Chroma collection with local embeddings."""

    def __init__(self, persist_directory: Optional[str] = None):
        kb_config = settings.kb
        self.persist_directory = persist_directory or kb_config["persist_directory"]
        self.collection_name = kb_config["collection_name"]
        self.embedding_model = kb_config["embedding_model"]

        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
        self.collection = self._get_or_create_collection()
        logger.info(
            "KnowledgeBase ready: collection='%s' (%d documents)",
            self.collection_name,
            self.collection.count(),
        )

    def _get_or_create_collection(self):
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

    # Ingestion: PDF → chunks → collection
    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Parse a PDF using the configured method and add chunks to the KB.
        """
        method = settings.parsing["method"]
        logger.info("Ingesting PDF '%s' with method '%s'", pdf_path, method)

        if method == "landing_ai":
            chunks = self._parse_with_landing_ai(pdf_path)
        elif method == "langchain":
            chunks = self._parse_with_langchain(pdf_path)
        else:
            raise KnowledgeBaseError(f"Unknown parsing method: {method}")

        if not chunks:
            logger.warning("No chunks extracted from PDF; nothing added.")
            return

        # Add to Chroma
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": os.path.basename(pdf_path), "chunk_index": i} for i in range(len(chunks))]
        self.add_chunks(chunks, metadatas, ids)
        logger.info("Indexed %d chunks from PDF.", len(chunks))

    def _parse_with_landing_ai(self, pdf_path: str) -> List[str]:
        """
        Call Landing AI Document Pretrained Transformer API.
        Expects API key in environment variable LANDING_AI_API_KEY
        and endpoint in settings.parsing.landing_ai.endpoint.
        Returns list of text chunks.
        """
        # Placeholder for actual API call
        import requests

        api_key = os.environ.get("LANDING_AI_API_KEY")
        if not api_key:
            raise KnowledgeBaseError("Landing AI API key not set (LANDING_AI_API_KEY).")

        endpoint = settings.parsing["landing_ai"]["endpoint"]
        with open(pdf_path, "rb") as f:
            files = {"file": f}
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.post(endpoint, files=files, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

        # Extract text blocks from Landing AI response (structure depends on API)
        # Assuming response contains a list of parsed elements with 'text'
        blocks = data.get("text_blocks", [])
        chunks = [block.get("text", "") for block in blocks if block.get("text")]
        logger.info("Landing AI returned %d text blocks.", len(chunks))
        return chunks

    def _parse_with_langchain(self, pdf_path: str) -> List[str]:
        """
        Use LangChain's PyPDFLoader + RecursiveCharacterTextSplitter.
        Returns list of text chunks.
        """
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info("PyPDFLoader extracted %d pages.", len(documents))

        chunk_size = settings.parsing["chunk_size"]
        chunk_overlap = settings.parsing["chunk_overlap"]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        chunk_texts = [doc.page_content for doc in chunks]
        logger.info("RecursiveCharacterTextSplitter produced %d chunks.", len(chunk_texts))
        return chunk_texts

    # Manual chunk ingestion (for dummy text etc.)
    def add_chunks(
        self,
        chunks: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if not chunks:
            return
        if ids is None:
            ids = [str(i) for i in range(len(chunks))]
        if metadatas is None:
            metadatas = [{}] * len(chunks)
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        logger.info("Added %d chunks (total now %d)", len(chunks), self.count())

    # Retrieval
    def query(self, query_text: str, k: Optional[int] = None) -> List[Dict]:
        if self.count() == 0:
            raise KnowledgeBaseEmptyError(self.collection_name)
        k = k or settings.retrieval["k"]
        try:
            results = self.collection.query(query_texts=[query_text], n_results=k)
        except Exception as exc:
            logger.exception("Query failed: %s", query_text)
            raise KnowledgeBaseQueryError(query_text, str(exc)) from exc

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"description": "Dermatology knowledge base"},
        )



# Convenience: create KB from raw text (dummy KB or pre‑parsed)
def create_kb_from_text(text: str) -> KnowledgeBase:
    """Split a text using RecursiveCharacterTextSplitter (same as langchain method)
    and index into a fresh KnowledgeBase."""
    kb = KnowledgeBase()
    kb.clear()

    chunk_size = settings.parsing["chunk_size"]
    chunk_overlap = settings.parsing["chunk_overlap"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    kb.add_chunks(chunks, [{"source": "dummy_kb"}] * len(chunks))
    logger.info("Created KB from dummy text: %d chunks", len(chunks))
    return kb