"""
Agent models and schemas for structured responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AIProvider(str, Enum):
    """Supported AI providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    COHERE = "cohere"
    MISTRAL = "mistral"


class AgentConfig(BaseModel):
    """Configuration for an AI agent."""

    name: str = Field(..., description="Agent name")
    provider: AIProvider = Field(default=AIProvider.OPENAI, description="AI provider")
    model: str = Field(..., description="Model name")
    system_prompt: str = Field(..., description="System prompt for the agent")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens in response")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")
    tools: list[str] = Field(default_factory=list, description="list of available tool names")


class AgentMessage(BaseModel):
    """A message in the conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Message ID")
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ToolCall(BaseModel):
    """A tool call made by the agent."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Tool call ID")
    name: str = Field(..., description="Tool name")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Tool call timestamp")


class ToolResult(BaseModel):
    """Result of a tool execution."""

    call_id: str = Field(..., description="Tool call ID")
    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Tool result timestamp")


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by agents."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")
    required: list[str] = Field(default_factory=list, description="Required parameters")


class AgentResponse(BaseModel):
    """Response from an agent."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Response ID")
    agent_name: str = Field(..., description="Name of the agent that generated the response")
    content: str = Field(..., description="Response content")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls made during response generation")
    tool_results: list[ToolResult] = Field(default_factory=list, description="Results of tool executions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    usage: Optional[dict[str, Any]] = Field(default=None, description="Token usage information")


class AgentDependencies(BaseModel):
    """Dependencies that can be injected into agent context."""

    user_id: Optional[str] = Field(default=None, description="User ID for the current session")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    tools: dict[str, Any] = Field(default_factory=dict, description="Available tools")


class ConversationHistory(BaseModel):
    """Conversation history for an agent session."""

    session_id: str = Field(..., description="Session ID")
    agent_name: str = Field(..., description="Agent name")
    messages: list[AgentMessage] = Field(default_factory=list, description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Conversation creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class AgentStatus(str, Enum):
    """Agent status."""

    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class AgentInstance(BaseModel):
    """An instance of an agent."""

    id: str = Field(default_factory=lambda: str(uuid4()), description="Agent instance ID")
    config: AgentConfig = Field(..., description="Agent configuration")
    status: AgentStatus = Field(default=AgentStatus.IDLE, description="Agent status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last used timestamp")
    usage_count: int = Field(default=0, description="Number of times this agent has been used")
