import openai
from sqlmodel import Session, select

from src.agents import models as ai_models
from src.agents.ai_client import AIClient


class AIService:
    @staticmethod
    async def create_chat_completion(openai_client: openai.OpenAI, messages: list[dict]) -> str:
        response = await AIClient.create_completion(openai_client, messages)
        return AIClient.parse_response(response)

    @staticmethod
    def create_conversation(db: Session, user_id: int, title: str):
        conversation = ai_models.Conversation(user_id=user_id, title=title)
        db.add(conversation)
        db.commit()
        return conversation

    @staticmethod
    def add_message(db: Session, conversation_id: int, role: str, content: str):
        message = ai_models.Message(conversation_id=conversation_id, role=role, content=content)
        db.add(message)
        db.commit()
        return message

    @staticmethod
    def user_belongs_to_conversation(db: Session, user_id: int, conversation_id: int) -> bool:
        statement = select(ai_models.Conversation).where(
            ai_models.Conversation.id == conversation_id,
            ai_models.Conversation.user_id == user_id
        )
        conversation = db.exec(statement).first()
        return conversation is not None
