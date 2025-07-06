from collections.abc import Generator
from functools import lru_cache
from typing import Annotated

import openai
from fastapi import Depends
from sqlmodel import Session

from src.config import settings
from src.db import engine


@lru_cache
def get_openai_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY)


def get_db() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]
OpenAIClient = Annotated[openai.OpenAI, Depends(get_openai_client)]
