import openai

from src.config import settings


class AIClient:
    @staticmethod
    async def create_completion(client: openai.OpenAI, messages: list[dict]):
        return await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=settings.OPENAI_MAX_TOKENS,
        )

    @staticmethod
    def parse_response(response) -> str:
        return response.choices[0].message.content
