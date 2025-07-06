from fastapi import APIRouter, HTTPException

from src.chat import schemas as ai_schemas
from src.chat.dependencies import DbSession, OpenAIClient
from src.chat.services import AIService
from src.auth.dependencies import CurrentUser

router = APIRouter(prefix="/ai", tags=["ai"])


@router.post("/conversations/", response_model=ai_schemas.Conversation)
async def create_conversation(
    data: ai_schemas.ConversationCreate,
    db: DbSession,
    current_user: CurrentUser,
):
    return AIService.create_conversation(db=db, user_id=current_user.id, title=data.title)


@router.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: int,
    message: ai_schemas.MessageCreate,
    db: DbSession,
    openai_client: OpenAIClient,
    current_user: CurrentUser,
):
    if not AIService.user_belongs_to_conversation(db, current_user.id, conversation_id):
        raise HTTPException(status_code=403, detail="User does not have access to this conversation")

    user_message = AIService.add_message(db=db, conversation_id=conversation_id, role="user", content=message.content)
    response = await AIService.create_chat_completion(openai_client, [{"role": "user", "content": message.content}])
    ai_message = AIService.add_message(db=db, conversation_id=conversation_id, role="assistant", content=response)

    return {"user_message": user_message, "assistant_message": ai_message}
