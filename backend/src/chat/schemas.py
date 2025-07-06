from pydantic import BaseModel


class MessageBase(BaseModel):
    role: str
    content: str
    metadata: dict | None = {}


class MessageCreate(MessageBase):
    pass


class Message(MessageBase):
    id: int
    conversation_id: int

    class Config:
        from_attributes = True


class ConversationCreate(BaseModel):
    title: str


class Conversation(BaseModel):
    id: int
    title: str
    messages: list[Message]

    class Config:
        from_attributes = True
