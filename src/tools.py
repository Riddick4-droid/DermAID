"""
LangChain tool definitions for the DermAId agent.
"""

import re
from typing import Optional

from langchain.tools import tool

from .logger import logger
from .exceptions import AgentToolError
from .config import settings
from .knowledge_base import KnowledgeBase
from .vision import classifier  # may be None if model failed to load


# Lazy-loaded KnowledgeBase instance (singleton)
_kb_instance: Optional[KnowledgeBase] = None

def _get_kb() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        logger.debug("Lazy-initialised KnowledgeBase singleton.")
    return _kb_instance



# Tool 1: Simple Retrieval
@tool
def simple_retrieve(query: str) -> str:
    """
    Retrieve relevant medical information from the dermatology knowledge base
    for a single condition or general query.
    Input: a string query (e.g., 'acne treatment').
    Returns: top-k most relevant chunks with metadata.
    """
    try:
        kb = _get_kb()
        results = kb.query(query)
        if not results:
            return "No relevant information found in the knowledge base."

        output_lines = []
        for i, res in enumerate(results):
            text = res["text"].strip()
            meta = res.get("metadata", {})
            condition = meta.get("condition", "Unknown")
            output_lines.append(
                f"--- Result {i+1} (Condition: {condition}) ---\n{text}"
            )
        return "\n\n".join(output_lines)
    except Exception as exc:
        logger.exception("simple_retrieve failed for query '%s'", query)
        raise AgentToolError("simple_retrieve", str(exc)) from exc



# Tool 2: Image Analysis
@tool
def analyze_image(image_path: str) -> str:
    """
    Analyze a skin image and return predicted condition with confidence.
    Input: path to image file (string).
    Returns: a sentence like 'Predicted: acne vulgaris (confidence: 0.95)'
             or 'Uncertain prediction: ...' if confidence below threshold.
    """
    try:
        if classifier is None:
            return "Vision model is not available (failed to load)."

        pred, conf = classifier.predict(image_path)
        threshold = settings.vision["confidence_threshold"]
        if conf >= threshold:
            return f"Predicted: {pred} (confidence: {conf:.2f})"
        else:
            return f"Uncertain prediction: {pred} (confidence: {conf:.2f}). Image features unclear."
    except Exception as exc:
        logger.exception("analyze_image failed for path '%s'", image_path)
        raise AgentToolError("analyze_image", str(exc)) from exc


# Tool 3: Compare Conditions (powered by LlamaIndex)
@tool
def compare_conditions(query: str) -> str:
    """
    Compare two skin conditions. Input must explicitly mention two conditions,
    e.g., 'compare acne and rosacea' or 'difference between eczema and psoriasis'.
    Returns: combined relevant evidence for both conditions.
    """
    try:
        kb = _get_kb()
        # Extract two condition names from the query
        pattern = r"(?:compare|difference between)\s+([a-zA-Z\s]+?)\s+(?:and|vs|or)\s+([a-zA-Z\s]+?)(?:\s*\?|$)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            # fallback: split by ' and ' or ' vs '
            cleaned = re.sub(r'compare|difference between', '', query, flags=re.IGNORECASE).strip()
            parts = re.split(r'\s+(?:and|vs|or)\s+', cleaned)
            if len(parts) < 2:
                return "Could not identify two conditions to compare. Please rephrase like 'compare X and Y'."
            cond1 = parts[0].strip()
            cond2 = parts[1].strip().rstrip('?')
        else:
            cond1 = match.group(1).strip()
            cond2 = match.group(2).strip().rstrip('?')

        # Retrieve for each condition using Chroma directly (could also use LlamaIndex sub-queries)
        res1 = kb.query(cond1)
        res2 = kb.query(cond2)

        def _format(condition: str, results: list) -> str:
            if not results:
                return f"No information found for {condition}."
            lines = []
            for r in results:
                snippet = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
                lines.append(f"  [source: {r['metadata'].get('condition', 'unknown')}] {snippet}")
            return "\n".join(lines)

        out = f"Evidence for **{cond1.title()}**:\n{_format(cond1, res1)}\n\n"
        out += f"Evidence for **{cond2.title()}**:\n{_format(cond2, res2)}"
        return out

    except Exception as exc:
        logger.exception("compare_conditions failed for query '%s'", query)
        raise AgentToolError("compare_conditions", str(exc)) from exc