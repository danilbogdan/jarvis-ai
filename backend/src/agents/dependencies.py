"""
Dependency injection for agent context and services.
"""

from typing import Optional

from fastapi import Depends

from .models import AgentDependencies
from .services import AgentService, agent_service


def get_agent_service() -> AgentService:
    """Get the global agent service instance."""
    return agent_service


def get_agent_dependencies(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentDependencies:
    """Create agent dependencies for injection."""
    return AgentDependencies(
        user_id=user_id,
        session_id=session_id,
    )
