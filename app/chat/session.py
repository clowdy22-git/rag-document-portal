"""
Chat session memory. Each session holds a message history so multi-turn
conversations (follow-up questions, "what about X instead?") work correctly.

This is in-memory only — sessions are lost on process restart. That's fine
for local development; Phase 6 (AWS deployment) can swap this for Redis or
DynamoDB without changing the interface below.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ChatSession:
    session_id: str
    document_ids: list[str] = field(default_factory=list)  # which docs this session can query
    messages: list[ChatMessage] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(ChatMessage(role="assistant", content=content))

    def recent_history(self, max_turns: int = 5) -> list[ChatMessage]:
        """Return the last N turns (a turn = one user + one assistant message)."""
        return self.messages[-(max_turns * 2):]

    def history_as_text(self, max_turns: int = 5) -> str:
        """Format recent history as plain text for prompting."""
        lines = []
        for msg in self.recent_history(max_turns):
            speaker = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{speaker}: {msg.content}")
        return "\n".join(lines)