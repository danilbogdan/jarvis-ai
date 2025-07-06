"""
Agent-based AI system using PydanticAI.

This module provides a provider-agnostic AI agent system with tool support,
dependency injection, and structured responses.
"""

from .ai_client import AIClient
from .dependencies import get_agent_dependencies, get_agent_service
from .models import (
    AgentConfig,
    AgentDependencies,
    AgentMessage,
    AgentResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .services import AgentService
from .tools import tool_registry, tool
from .schemas import (
    ChatRequest,
    ChatResponse,
    CreateAgentRequest,
    UpdateAgentRequest,
)

__all__ = [
    "AgentConfig",
    "AgentResponse",
    "AgentMessage",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "AgentDependencies",
    "AIClient",
    "AgentService",
    "get_agent_service",
    "get_agent_dependencies",
    "tool_registry",
    "tool",
    "ChatRequest",
    "ChatResponse",
    "CreateAgentRequest",
    "UpdateAgentRequest",
]
