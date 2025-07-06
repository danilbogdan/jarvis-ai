import uuid
from typing import TYPE_CHECKING, Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel

if TYPE_CHECKING:
    from src.auth.models import User


class Message(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    role: str
    content: str
    meta: dict | None = Field(default=None, sa_column=Column(JSON))

    conversation: "Conversation" = Relationship(back_populates="messages")


class Conversation(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id")
    title: str

    messages: list[Message] = Relationship(back_populates="conversation", cascade_delete=True)
    user: Optional["User"] = Relationship(back_populates="conversations")
