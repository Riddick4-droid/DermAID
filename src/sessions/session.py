"""
Persistent per‑session chat history stored as JSON files.
Implements langchain_core.chat_history.BaseChatMessageHistory
and plugs directly into ConversationBufferWindowMemory.
"""

import json
import os
import fcntl
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.messages import messages_from_dict, messages_to_dict

from src.logger import logger
from src.exceptions import SessionNotFoundError, SessionStorageError


class JSONChatMessageHistory(BaseChatMessageHistory):
    """Thread‑safe, file‑backed chat history for a single session."""

    def __init__(self, session_id: str, directory: str = "sessions"):
        self.session_id = session_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)#save in the same directory as the script
        self.file_path = self.directory / f"{session_id}.json"
        self._ensure_file()


    @property
    def messages(self) -> List[BaseMessage]:
        """Return all messages in the session."""
        data = self._load()
        messages = messages_from_dict(data.get("history", []))
        logger.debug("Loaded %d messages for session %s", len(messages), self.session_id)
        return messages

    def add_user_message(self, message: str) -> None:
        """Append a user message to the history."""
        self._add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Append an AI message to the history."""
        self._add_message(AIMessage(content=message))

    def clear(self) -> None:
        """Delete all history for this session."""
        if self.file_path.exists():
            self.file_path.unlink()
            logger.info("Cleared history for session %s", self.session_id)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create the JSON file if it doesn't exist."""
        if not self.file_path.exists():
            initial = {
                "session_id": self.session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "history": [],
            }
            try:
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump(initial, f, indent=2)
                logger.info("Created new session file: %s", self.file_path)
            except OSError as exc:
                raise SessionStorageError(self.session_id, str(exc)) from exc

    def _load(self) -> dict:
        """Read the JSON file with shared lock."""
        if not self.file_path.exists():
            raise SessionNotFoundError(self.session_id)
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
            return data
        except (OSError, json.JSONDecodeError) as exc:
            logger.exception("Failed to load session %s", self.session_id)
            raise SessionStorageError(self.session_id, str(exc)) from exc

    def _save(self, data: dict) -> None:
        """Write the JSON file atomically using a temporary file."""
        tmp_path = self.file_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(data, f, indent=2, ensure_ascii=False)
                fcntl.flock(f, fcntl.LOCK_UN)
            os.replace(tmp_path, self.file_path)  # atomic on Unix
            logger.debug("Saved session %s", self.session_id)
        except OSError as exc:
            logger.exception("Failed to save session %s", self.session_id)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise SessionStorageError(self.session_id, str(exc)) from exc

    def _add_message(self, message: BaseMessage) -> None:
        """Append a message to the history and persist."""
        data = self._load()
        msg_dict = messages_to_dict([message])[0]
        msg_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        data["history"].append(msg_dict)
        self._save(data)

    def __repr__(self) -> str:
        return f"<JSONChatMessageHistory session='{self.session_id}'>"