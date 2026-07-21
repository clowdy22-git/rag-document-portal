"""
Manages multiple concurrent chat sessions in memory, keyed by session_id.
"""

import uuid
from app.chat.session import ChatSession


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ChatSession] = {}

    def create_session(self, document_ids: list[str] | None = None) -> ChatSession:
        session_id = uuid.uuid4().hex[:12]
        session = ChatSession(session_id=session_id, document_ids=document_ids or [])
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None, document_ids: list[str] | None = None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(document_ids=document_ids)