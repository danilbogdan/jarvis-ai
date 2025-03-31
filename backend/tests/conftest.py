import uuid
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, delete, select

from src.agents.models import Conversation, Message
from src.agents.services import AIService
from src.auth.models import Item, User
from src.config import settings
from src.db import engine, init_db
from src.main import app
from tests.auth.utils.user import authentication_token_from_email
from tests.utils import get_superuser_token_headers, random_lower_string


@pytest.fixture(scope="module")
def superuser_id(db: Session) -> uuid.UUID:
    """Get the superuser ID from the database"""
    statement = select(User).where(User.email == settings.FIRST_SUPERUSER)
    superuser = db.exec(statement).first()
    return superuser.id


@pytest.fixture(scope="module")
def conversation(db: Session, superuser_id: uuid.UUID) -> Conversation:
    """Create a test conversation for the superuser"""
    return AIService.create_conversation(db=db, user_id=superuser_id, title=random_lower_string())


@pytest.fixture(scope="module")
def message(db: Session, conversation: Conversation) -> Message:
    """Create a test message in the conversation"""
    return AIService.add_message(db=db, conversation_id=conversation.id, role="user", content=random_lower_string())


@pytest.fixture(scope="module")
def mock_openai_response():
    """Mock OpenAI API response"""

    class MockChoice:
        def __init__(self, content):
            self.message = type("Message", (), {"content": content})

    class MockResponse:
        def __init__(self, content="Mock AI response"):
            self.choices = [MockChoice(content)]

    return MockResponse()


@pytest.fixture(scope="module")
def mock_openai_client(monkeypatch, mock_openai_response):
    """Mock OpenAI client for testing"""

    class MockOpenAI:
        class ChatCompletion:
            @staticmethod
            async def create(*args, **kwargs):
                return mock_openai_response

        chat = ChatCompletion()

    return MockOpenAI()


@pytest.fixture(scope="module")
def conversation_data():
    """Test conversation data"""
    return {"title": random_lower_string()}


@pytest.fixture(scope="module")
def message_data():
    """Test message data"""
    return {"role": "user", "content": random_lower_string(), "metadata": None}


@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> dict[str, str]:
    return get_superuser_token_headers(client)


@pytest.fixture(scope="module")
def normal_user_token_headers(client: TestClient, db: Session) -> dict[str, str]:
    return authentication_token_from_email(client=client, email=settings.EMAIL_TEST_USER, db=db)


@pytest.fixture(scope="session", autouse=True)
def db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        init_db(session)
        yield session
        # Clean up agents tables first due to foreign key constraints
        statement = delete(Message)
        session.exec(statement)
        statement = delete(Conversation)
        session.exec(statement)
        # Then clean up auth tables
        statement = delete(Item)
        session.exec(statement)
        statement = delete(User)
        session.exec(statement)
        session.commit()


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c
