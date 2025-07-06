from fastapi.testclient import TestClient

from src.config import settings
from tests.utils import random_lower_string


def test_create_conversation(
    client: TestClient, superuser_token_headers: dict[str, str], conversation_data: dict
) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/ai/conversations/",
        headers=superuser_token_headers,
        json=conversation_data,
    )
    assert r.status_code == 200
    created_conversation = r.json()
    assert created_conversation["title"] == conversation_data["title"]
    assert "id" in created_conversation
    assert "messages" in created_conversation
    assert created_conversation["messages"] == []


def test_chat(
    client: TestClient,
    superuser_token_headers: dict[str, str],
    conversation: dict,
    message_data: dict,
) -> None:
    r = client.post(
        f"{settings.API_V1_STR}/ai/conversations/{conversation.id}/chat",
        headers=superuser_token_headers,
        json=message_data,
    )
    assert r.status_code == 200
    response = r.json()
    assert "user_message" in response
    assert "assistant_message" in response
    assert response["user_message"]["content"] == message_data["content"]
    assert response["user_message"]["role"] == "user"
    assert response["assistant_message"]["role"] == "assistant"
    assert len(response["assistant_message"]["content"]) > 0


def test_chat_unauthorized_conversation(
    client: TestClient, superuser_token_headers: dict[str, str], normal_user_token_headers: dict[str, str]
) -> None:
    # Create conversation as superuser
    title = random_lower_string()
    conversation = client.post(
        f"{settings.API_V1_STR}/ai/conversations/",
        headers=superuser_token_headers,
        json={"title": title},
    ).json()

    # Try to chat as normal user
    data = {"role": "user", "content": random_lower_string(), "metadata": None}
    r = client.post(
        f"{settings.API_V1_STR}/ai/conversations/{conversation['id']}/chat",
        headers=normal_user_token_headers,
        json=data,
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "User does not have access to this conversation"
