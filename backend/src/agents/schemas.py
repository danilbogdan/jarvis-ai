"""
Pydantic schemas for agent API requests and responses.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models import AIProvider


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""

    message: str = Field(..., description="User message to send to the agent")
    session_id: str | None = Field(default=None, description="Session ID for conversation continuity")
    agent_name: str | None = Field(default="general_assistant", description="Name of the agent to use")
    context: dict[str, Any] | None = Field(default=None, description="Additional context for the agent")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""

    response: str = Field(..., description="Agent's response")
    agent_name: str = Field(..., description="Name of the agent that responded")
    session_id: str | None = Field(default=None, description="Session ID")
    tool_calls: list[dict[str, Any]] = Field(default_factory=list, description="Tool calls made by the agent")
    tool_results: list[dict[str, Any]] = Field(default_factory=list, description="Results of tool executions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class StreamChatRequest(BaseModel):
    """Request schema for streaming chat endpoint."""

    message: str = Field(..., description="User message to send to the agent")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    agent_name: Optional[str] = Field(default="general_assistant", description="Name of the agent to use")
    context: Optional[dict[str, Any]] = Field(default=None, description="Additional context for the agent")


class CreateAgentRequest(BaseModel):
    """Request schema for creating a new agent."""

    name: str = Field(..., description="Agent name")
    provider: AIProvider = Field(default=AIProvider.OPENAI, description="AI provider")
    model: str = Field(..., description="Model name")
    system_prompt: str = Field(..., description="System prompt for the agent")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: int = Field(default=2000, gt=0, description="Maximum tokens in response")
    timeout: int = Field(default=30, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries")
    tools: list[str] = Field(default_factory=list, description="list of available tool names")


class UpdateAgentRequest(BaseModel):
    """Request schema for updating an agent."""

    provider: Optional[AIProvider] = Field(default=None, description="AI provider")
    model: Optional[str] = Field(default=None, description="Model name")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the agent")
    temperature: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="Temperature for response generation"
    )
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens in response")
    timeout: Optional[int] = Field(default=None, gt=0, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(default=None, ge=0, description="Maximum number of retries")
    tools: Optional[list[str]] = Field(default=None, description="list of available tool names")


class AgentResponse(BaseModel):
    """Response schema for agent operations."""

    id: str = Field(..., description="Agent instance ID")
    name: str = Field(..., description="Agent name")
    provider: str = Field(..., description="AI provider")
    model: str = Field(..., description="Model name")
    status: str = Field(..., description="Agent status")
    usage_count: int = Field(..., description="Number of times this agent has been used")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_used: Optional[datetime] = Field(default=None, description="Last used timestamp")
    tools: list[str] = Field(default_factory=list, description="Available tools")


class AgentListResponse(BaseModel):
    """Response schema for listing agents."""

    agents: list[AgentResponse] = Field(..., description="list of agents")
    total: int = Field(..., description="Total number of agents")


class AgentStatsResponse(BaseModel):
    """Response schema for agent statistics."""

    name: str = Field(..., description="Agent name")
    provider: str = Field(..., description="AI provider")
    model: str = Field(..., description="Model name")
    status: str = Field(..., description="Agent status")
    usage_count: int = Field(..., description="Number of times this agent has been used")
    created_at: str = Field(..., description="Creation timestamp")
    last_used: str | None = Field(default=None, description="Last used timestamp")
    tools: list[str] = Field(default_factory=list, description="Available tools")


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics."""

    total_agents: int = Field(..., description="Total number of agents")
    agents: dict[str, AgentStatsResponse] = Field(..., description="Agent statistics")
    total_conversations: int = Field(..., description="Total number of conversations")
    available_tools: list[str] = Field(..., description="Available tools")


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""

    session_id: str = Field(..., description="Session ID")
    agent_name: str = Field(..., description="Agent name")
    messages: list[dict[str, Any]] = Field(..., description="Conversation messages")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ToolDefinitionResponse(BaseModel):
    """Response schema for tool definitions."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(..., description="Tool parameters schema")
    required: list[str] = Field(..., description="Required parameters")


class ToolListResponse(BaseModel):
    """Response schema for listing tools."""

    tools: list[ToolDefinitionResponse] = Field(..., description="list of available tools")
    total: int = Field(..., description="Total number of tools")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    code: Optional[int] = Field(default=None, description="Error code")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    agents_count: int = Field(..., description="Number of active agents")
    uptime: Optional[str] = Field(default=None, description="Service uptime")
