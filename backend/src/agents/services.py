"""
Agent service for managing agent instances and orchestrating AI interactions.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel

from ..config import settings
from .ai_client import AIClient, create_ai_client
from .models import (
    AgentConfig,
    AgentDependencies,
    AgentInstance,
    AgentMessage,
    AgentResponse,
    AgentStatus,
    AIProvider,
    ConversationHistory,
)
from .tools import ToolRegistry, tool_registry

logger = logging.getLogger(__name__)


class AgentService:
    """Service for managing AI agents and their interactions."""

    def __init__(self, tool_registry: ToolRegistry = tool_registry):
        self.agents: dict[str, AgentInstance] = {}
        self.clients: dict[str, AIClient] = {}
        self.conversations: dict[str, ConversationHistory] = {}
        self.tool_registry = tool_registry
        self._initialize_default_agents()

    def _initialize_default_agents(self) -> None:
        """Initialize default agents based on configuration."""
        default_configs = [
            AgentConfig(
                name="general_assistant",
                provider=AIProvider.OPENAI,
                model=settings.OPENAI_MODEL,
                system_prompt="You are a helpful AI assistant. Use the available tools to help users with their tasks.",
                temperature=settings.AGENT_TEMPERATURE,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                timeout=settings.AGENT_TIMEOUT,
                max_retries=settings.AGENT_MAX_RETRIES,
                tools=["get_current_time", "calculate", "make_http_request"],
            ),
            AgentConfig(
                name="code_assistant",
                provider=AIProvider.OPENAI,
                model=settings.OPENAI_MODEL,
                system_prompt="You are a specialized coding assistant. Help users with programming tasks, code review, and technical questions. Use file system tools when needed.",
                temperature=0.3,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                timeout=settings.AGENT_TIMEOUT,
                max_retries=settings.AGENT_MAX_RETRIES,
                tools=["read_file", "write_file", "list_directory", "run_command"],
            ),
            AgentConfig(
                name="research_assistant",
                provider=AIProvider.OPENAI,
                model=settings.OPENAI_MODEL,
                system_prompt="You are a research assistant. Help users gather information, analyze data, and provide insights. Use web tools to fetch current information.",
                temperature=0.5,
                max_tokens=settings.OPENAI_MAX_TOKENS,
                timeout=settings.AGENT_TIMEOUT,
                max_retries=settings.AGENT_MAX_RETRIES,
                tools=["make_http_request", "get_current_time", "calculate"],
            ),
        ]

        for config in default_configs:
            self.create_agent(config)

    def create_agent(self, config: AgentConfig) -> AgentInstance:
        """Create a new agent instance."""
        if config.name in self.agents:
            raise ValueError(f"Agent '{config.name}' already exists")

        # Create AI client
        client = create_ai_client(config)

        # Get tools for this agent
        tools = []
        for tool_name in config.tools:
            tool_func = self.tool_registry.get_tool(tool_name)
            if tool_func:
                tools.append(tool_func)
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry")

        # Create PydanticAI agent
        agent = client.create_agent(
            deps_type=AgentDependencies,
            tools=tools,
        )

        # Create agent instance
        instance = AgentInstance(
            config=config,
            status=AgentStatus.IDLE,
        )

        self.agents[config.name] = instance
        self.clients[config.name] = client

        logger.info(f"Created agent: {config.name}")
        return instance

    def get_agent(self, name: str) -> Optional[AgentInstance]:
        """Get an agent instance by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """list all agent names."""
        return list(self.agents.keys())

    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """Get agent status."""
        agent = self.agents.get(name)
        return agent.status if agent else None

    def update_agent_config(self, name: str, config: AgentConfig) -> bool:
        """Update agent configuration."""
        if name not in self.agents:
            return False

        # Update the instance
        self.agents[name].config = config

        # Recreate the client with new config
        client = create_ai_client(config)

        # Get tools for this agent
        tools = []
        for tool_name in config.tools:
            tool_func = self.tool_registry.get_tool(tool_name)
            if tool_func:
                tools.append(tool_func)

        # Recreate PydanticAI agent
        agent = client.create_agent(
            deps_type=AgentDependencies,
            tools=tools,
        )

        self.clients[name] = client
        logger.info(f"Updated agent: {name}")
        return True

    def delete_agent(self, name: str) -> bool:
        """Delete an agent."""
        if name not in self.agents:
            return False

        del self.agents[name]
        del self.clients[name]

        # Clean up conversations
        conversations_to_remove = [
            session_id for session_id, conv in self.conversations.items() if conv.agent_name == name
        ]
        for session_id in conversations_to_remove:
            del self.conversations[session_id]

        logger.info(f"Deleted agent: {name}")
        return True

    async def chat_with_agent(
        self,
        agent_name: str,
        message: str,
        session_id: Optional[str] = None,
        dependencies: Optional[AgentDependencies] = None,
    ) -> AgentResponse:
        """Chat with an agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent_instance = self.agents[agent_name]
        client = self.clients[agent_name]

        # Update agent status
        agent_instance.status = AgentStatus.PROCESSING

        try:
            # Get or create conversation history
            if session_id:
                conversation = self.conversations.get(session_id)
                if not conversation:
                    conversation = ConversationHistory(
                        session_id=session_id,
                        agent_name=agent_name,
                    )
                    self.conversations[session_id] = conversation
            else:
                conversation = None

            # Prepare message history for the AI
            message_history = []
            if conversation:
                for msg in conversation.messages:
                    message_history.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

            # Create dependencies if not provided
            if not dependencies:
                dependencies = AgentDependencies(
                    session_id=session_id,
                    context={"agent_name": agent_name},
                    tools=self.tool_registry.get_all_definitions(),
                )

            # Run the agent
            response = await client.run_agent(
                message=message,
                deps=dependencies,
                message_history=message_history,
            )

            # Update conversation history
            if conversation:
                # Add user message
                conversation.messages.append(
                    AgentMessage(
                        role="user",
                        content=message,
                    )
                )

                # Add assistant response
                conversation.messages.append(
                    AgentMessage(
                        role="assistant",
                        content=response.content,
                        metadata={
                            "tool_calls": [call.model_dump() for call in response.tool_calls],
                            "tool_results": [result.model_dump() for result in response.tool_results],
                        },
                    )
                )

                conversation.updated_at = datetime.utcnow()

            # Update agent instance
            agent_instance.status = AgentStatus.IDLE
            agent_instance.last_used = datetime.utcnow()
            agent_instance.usage_count += 1

            return response

        except Exception as e:
            agent_instance.status = AgentStatus.ERROR
            logger.error(f"Error in agent '{agent_name}': {str(e)}")
            raise

    async def stream_chat_with_agent(
        self,
        agent_name: str,
        message: str,
        session_id: Optional[str] = None,
        dependencies: Optional[AgentDependencies] = None,
    ):
        """Stream chat with an agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent_instance = self.agents[agent_name]
        client = self.clients[agent_name]

        # Update agent status
        agent_instance.status = AgentStatus.PROCESSING

        try:
            # Get or create conversation history
            if session_id:
                conversation = self.conversations.get(session_id)
                if not conversation:
                    conversation = ConversationHistory(
                        session_id=session_id,
                        agent_name=agent_name,
                    )
                    self.conversations[session_id] = conversation
            else:
                conversation = None

            # Prepare message history for the AI
            message_history = []
            if conversation:
                for msg in conversation.messages:
                    message_history.append(
                        {
                            "role": msg.role,
                            "content": msg.content,
                        }
                    )

            # Create dependencies if not provided
            if not dependencies:
                dependencies = AgentDependencies(
                    session_id=session_id,
                    context={"agent_name": agent_name},
                    tools=self.tool_registry.get_all_definitions(),
                )

            # Stream the agent response
            async for chunk in client.stream_agent(
                message=message,
                deps=dependencies,
                message_history=message_history,
            ):
                yield chunk

            # Update agent instance
            agent_instance.status = AgentStatus.IDLE
            agent_instance.last_used = datetime.utcnow()
            agent_instance.usage_count += 1

        except Exception as e:
            agent_instance.status = AgentStatus.ERROR
            logger.error(f"Error streaming agent '{agent_name}': {str(e)}")
            raise

    def get_conversation_history(self, session_id: str) -> Optional[ConversationHistory]:
        """Get conversation history for a session."""
        return self.conversations.get(session_id)

    def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False

    def get_agent_stats(self, agent_name: str) -> Optional[dict[str, Any]]:
        """Get agent statistics."""
        agent = self.agents.get(agent_name)
        if not agent:
            return None

        return {
            "name": agent.config.name,
            "provider": agent.config.provider.value,
            "model": agent.config.model,
            "status": agent.status.value,
            "usage_count": agent.usage_count,
            "created_at": agent.created_at.isoformat(),
            "last_used": agent.last_used.isoformat() if agent.last_used else None,
            "tools": agent.config.tools,
        }

    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all agents."""
        return {
            "total_agents": len(self.agents),
            "agents": {name: self.get_agent_stats(name) for name in self.agents},
            "total_conversations": len(self.conversations),
            "available_tools": self.tool_registry.list_tools(),
        }

    def register_tool(self, tool_func: Callable, **kwargs) -> None:
        """Register a new tool."""
        self.tool_registry.register_tool(func=tool_func, **kwargs)

    def get_available_tools(self) -> list[str]:
        """Get list of available tools."""
        return self.tool_registry.list_tools()


# Global agent service instance
agent_service = AgentService()
