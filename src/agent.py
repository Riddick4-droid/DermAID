"""
DermAId agent orchestrator.
Combines LangChain agent, tools, memory, and strictness guardrail.
"""

import os
from datetime import date
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from .logger import logger
from .config import settings
from .exceptions import AgentExecutionError
from src.sessions.session import JSONChatMessageHistory
from .tools import simple_retrieve, compare_conditions, analyze_image



# LLM initialisation (singleton)
_llm = ChatOpenAI(
    model=settings.llm["model_name"],
    temperature=settings.llm["temperature"],
    max_tokens=settings.llm.get("max_tokens", 1024),
)
logger.info("LLM initialised: model=%s", settings.llm["model_name"])


# System prompt template
SYSTEM_PROMPT = """
You are DermAId, a medical AI assistant specialized in dermatology.
You have access to a knowledge base of clinical guidelines and the following tools:
- simple_retrieve: for general or single-condition queries.
- compare_conditions: for comparing two skin conditions.
- analyze_image: to predict a skin condition from an image.

Conversation history is provided. If the user mentions a new condition different from the previous topic, do not force a comparison unless asked. If asked to compare, use the compare_conditions tool.

Strictness level (0=infer, 1=verbatim): {strictness}
When strictness is 0, you may elaborate slightly on the retrieved evidence to make the answer more helpful, but never invent medical facts not in the knowledge base.
When strictness is 1, you must quote the evidence verbatim and not add any external knowledge. Cite the source condition.

Current date: {date}
"""



# Memory factory
def _get_session_memory(session_id: str):
    """Create a ConversationBufferWindowMemory backed by our JSON KV cache."""
    history = JSONChatMessageHistory(session_id=session_id, directory=settings.agent["session_dir"])
    window_size = settings.agent["session_window_size"]
    return ConversationBufferWindowMemory(
        chat_memory=history,
        k=window_size,
        return_messages=True,
    )



# Main agent function
def run_agent(
    query: str,
    session_id: str,
    strictness: Optional[float] = None,
    image_path: Optional[str] = None,
) -> dict:
    """
    Execute the DermAId agent for a given query and session.

    Args:
        query: user's text query.
        session_id: unique session identifier (UUID string).
        strictness: 0–1 grounding strictness; falls back to config default if None.
        image_path: optional path to a skin image file.

    Returns:
        dict with keys 'answer', 'session_id', and 'error' (if any).
    """
    if strictness is None:
        strictness = settings.agent["strictness_default"]

    try:
        # Prepare memory
        memory = _get_session_memory(session_id)

        # Augment query if image provided
        augmented_query = query
        if image_path:
            try:
                vision_result = analyze_image(image_path)
                augmented_query = f"{vision_result}\nUser query: {query}"
            except Exception as exc:
                logger.warning("Image analysis failed, proceeding without it: %s", exc)
                augmented_query = f"Image analysis failed ({exc}).\nUser query: {query}"

        # Build prompt with current strictness and date
        system_text = SYSTEM_PROMPT.format(
            strictness=strictness,
            date=date.today().isoformat(),
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        tools = [simple_retrieve, compare_conditions, analyze_image]

        agent = create_openai_tools_agent(_llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
        )

        result = agent_executor.invoke({
            "input": augmented_query,
            "chat_history": memory.chat_memory.messages,
        })

        # Persist the exchange in session JSON
        memory.chat_memory.add_user_message(query)  # store original query
        memory.chat_memory.add_ai_message(result["output"])

        logger.info("Agent answered session %s (strictness=%.1f)", session_id, strictness)
        return {"answer": result["output"], "session_id": session_id, "error": None}

    except Exception as exc:
        logger.exception("Agent execution failed for session %s", session_id)
        return {"answer": None, "session_id": session_id, "error": str(exc)}