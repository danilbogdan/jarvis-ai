import os
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool

from src.config import settings

# Use in-memory SQLite for tests
TEST_SQLALCHEMY_DATABASE_URL = "sqlite://"


def get_test_engine():
    return create_engine(
        TEST_SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def get_test_session() -> Generator[Session, None, None]:
    engine = get_test_engine()
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
