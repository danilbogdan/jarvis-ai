import uuid

from sqlmodel import Session

from src.agents.services import AIService
from tests.utils import random_lower_string


def test_create_conversation(db: Session, superuser_id: uuid.UUID, conversation_data: dict) -> None:
    conversation = AIService.create_conversation(db=db, user_id=superuser_id, title=conversation_data["title"])

    assert conversation.title == conversation_data["title"]
    assert conversation.user_id == superuser_id
    assert conversation.id is not None


def test_add_message(db: Session, conversation: dict, message_data: dict) -> None:
    message = AIService.add_message(
        db=db, conversation_id=conversation.id, role=message_data["role"], content=message_data["content"]
    )

    assert message.content == message_data["content"]
    assert message.role == message_data["role"]
    assert message.conversation_id == conversation.id
    assert message.id is not None


def test_user_belongs_to_conversation(db: Session, superuser_id: uuid.UUID) -> None:
    other_user_id = uuid.uuid4()

    # Create conversation for user
    conversation = AIService.create_conversation(db=db, user_id=superuser_id, title=random_lower_string())

    # Check access
    assert AIService.user_belongs_to_conversation(db, superuser_id, conversation.id) is True
    assert AIService.user_belongs_to_conversation(db, other_user_id, conversation.id) is False
