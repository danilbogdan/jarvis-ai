"""
Provider-agnostic AI client wrapper using PydanticAI.
"""

import asyncio
import logging
from typing import Any, Callable, Optional, Type

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models import (
        AnthropicModel,
        CohereModel,
        GoogleModel,
        GroqModel,
        MistralModel,
        OpenAIModel,
    )
except ImportError:
    # Fallback for development - these will be available when pydantic-ai is installed
    Agent = Any
    RunContext = Any
    AnthropicModel = Any
    CohereModel = Any
    GoogleModel = Any
    GroqModel = Any
    MistralModel = Any
    OpenAIModel = Any

from ..config import settings
from .models import (
    AgentConfig,
    AgentDependencies,
    AgentResponse,
    AIProvider,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger(__name__)


class AIClient:
    """Provider-agnostic AI client using PydanticAI."""

    def __init__(self, config: AgentConfig):
        """Initialize the AI client with configuration."""
        self.config = config
        self._agent: Optional[Agent] = None
        self._model = self._create_model()

    def _create_model(self) -> Any:
        """Create the appropriate model based on provider."""
        provider = self.config.provider
        model_name = self.config.model

        if provider == AIProvider.OPENAI:
            return OpenAIModel(
                model_name=model_name,
                api_key=settings.OPENAI_API_KEY,
            )
        elif provider == AIProvider.ANTHROPIC:
            return AnthropicModel(
                model_name=model_name,
                api_key=settings.ANTHROPIC_API_KEY,
            )
        elif provider == AIProvider.GOOGLE:
            return GoogleModel(
                model_name=model_name,
                api_key=settings.GOOGLE_API_KEY,
            )
        elif provider == AIProvider.GROQ:
            return GroqModel(
                model_name=model_name,
                api_key=settings.GROQ_API_KEY,
            )
        elif provider == AIProvider.COHERE:
            return CohereModel(
                model_name=model_name,
                api_key=settings.COHERE_API_KEY,
            )
        elif provider == AIProvider.MISTRAL:
            return MistralModel(
                model_name=model_name,
                api_key=settings.MISTRAL_API_KEY,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def create_agent(
        self,
        deps_type: Type[AgentDependencies] = AgentDependencies,
        output_type: Optional[Type] = None,
        tools: Optional[list[Callable]] = None,
    ) -> Agent:
        """Create a PydanticAI agent with the configured model."""

        agent = Agent(
            model=self._model,
            deps_type=deps_type,
            output_type=output_type,
            system_prompt=self.config.system_prompt,
        )

        # Register tools if provided
        if tools:
            for tool in tools:
                agent.tool(tool)

        self._agent = agent
        return agent

    async def run_agent(
        self,
        message: str,
        deps: Optional[AgentDependencies] = None,
        message_history: Optional[list[dict[str, str]]] = None,
    ) -> AgentResponse:
        """Run the agent with a message and return structured response."""

        if not self._agent:
            raise ValueError("Agent not created. Call create_agent() first.")

        try:
            # Convert message history to PydanticAI format if provided
            history = []
            if message_history:
                for msg in message_history:
                    history.append(msg)

            # Run the agent
            result = await self._agent.run(
                message,
                deps=deps,
                message_history=history,
            )

            # Extract tool calls and results from the result
            tool_calls = []
            tool_results = []

            # PydanticAI stores tool calls in the result's message history
            if hasattr(result, "all_messages"):
                for msg in result.all_messages():
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_calls.append(
                                ToolCall(
                                    name=tool_call.function.name,
                                    arguments=tool_call.function.arguments or {},
                                )
                            )

                    if hasattr(msg, "tool_results") and msg.tool_results:
                        for tool_result in msg.tool_results:
                            tool_results.append(
                                ToolResult(
                                    call_id=tool_result.call_id,
                                    success=not hasattr(tool_result, "error"),
                                    result=getattr(tool_result, "content", None),
                                    error=getattr(tool_result, "error", None),
                                )
                            )

            # Create response
            response = AgentResponse(
                agent_name=self.config.name,
                content=str(result.output),
                tool_calls=tool_calls,
                tool_results=tool_results,
                metadata={
                    "provider": self.config.provider.value,
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                },
                usage=getattr(result, "usage", None),
            )

            return response

        except Exception as e:
            logger.error(f"Error running agent {self.config.name}: {str(e)}")
            raise

    async def stream_agent(
        self,
        message: str,
        deps: Optional[AgentDependencies] = None,
        message_history: Optional[list[dict[str, str]]] = None,
    ):
        """Stream agent response."""

        if not self._agent:
            raise ValueError("Agent not created. Call create_agent() first.")

        try:
            # Convert message history to PydanticAI format if provided
            history = []
            if message_history:
                for msg in message_history:
                    history.append(msg)

            # Stream the agent response
            async with self._agent.run_stream(
                message,
                deps=deps,
                message_history=history,
            ) as stream:
                async for chunk in stream:
                    yield chunk

        except Exception as e:
            logger.error(f"Error streaming agent {self.config.name}: {str(e)}")
            raise

    def add_tool(self, tool_func: Callable) -> None:
        """Add a tool to the agent."""
        if not self._agent:
            raise ValueError("Agent not created. Call create_agent() first.")

        self._agent.tool(tool_func)

    def add_system_prompt(self, prompt_func: Callable) -> None:
        """Add a dynamic system prompt to the agent."""
        if not self._agent:
            raise ValueError("Agent not created. Call create_agent() first.")

        self._agent.system_prompt(prompt_func)

    @property
    def agent(self) -> Optional[Agent]:
        """Get the underlying PydanticAI agent."""
        return self._agent

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider.value


def create_ai_client(config: AgentConfig) -> AIClient:
    """Factory function to create an AI client."""
    return AIClient(config)
