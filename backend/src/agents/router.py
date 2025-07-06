"""
FastAPI router for agent endpoints.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .dependencies import get_agent_service
from .models import AgentConfig, AgentDependencies
from .schemas import (
    AgentListResponse,
    AgentResponse,
    AgentStatsResponse,
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    CreateAgentRequest,
    ErrorResponse,
    HealthResponse,
    StreamChatRequest,
    SystemStatsResponse,
    ToolDefinitionResponse,
    ToolListResponse,
    UpdateAgentRequest,
)
from .services import AgentService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def chat_with_agent(
    request: ChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> ChatResponse:
    """Chat with an AI agent."""
    try:
        # Create dependencies
        dependencies = AgentDependencies(
            session_id=request.session_id,
            context=request.context or {},
        )

        # Chat with agent
        response = await agent_service.chat_with_agent(
            agent_name=request.agent_name or "general_assistant",
            message=request.message,
            session_id=request.session_id,
            dependencies=dependencies,
        )

        return ChatResponse(
            response=response.content,
            agent_name=response.agent_name,
            session_id=request.session_id,
            tool_calls=[call.model_dump() for call in response.tool_calls],
            tool_results=[result.model_dump() for result in response.tool_results],
            metadata=response.metadata,
            timestamp=response.timestamp,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/chat/stream",
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def stream_chat_with_agent(
    request: StreamChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Stream chat with an AI agent."""
    try:
        # Create dependencies
        dependencies = AgentDependencies(
            session_id=request.session_id,
            context=request.context or {},
        )

        async def generate_stream():
            try:
                async for chunk in agent_service.stream_chat_with_agent(
                    agent_name=request.agent_name,
                    message=request.message,
                    session_id=request.session_id,
                    dependencies=dependencies,
                ):
                    yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                yield f"data: {{'error': '{str(e)}'}}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error in stream chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post(
    "/",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_agent(
    request: CreateAgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    """Create a new AI agent."""
    try:
        # Create agent config
        config = AgentConfig(
            name=request.name,
            provider=request.provider,
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=request.timeout,
            max_retries=request.max_retries,
            tools=request.tools,
        )

        # Create agent
        instance = agent_service.create_agent(config)

        return AgentResponse(
            id=instance.id,
            name=instance.config.name,
            provider=instance.config.provider.value,
            model=instance.config.model,
            status=instance.status.value,
            usage_count=instance.usage_count,
            created_at=instance.created_at,
            last_used=instance.last_used,
            tools=instance.config.tools,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/",
    response_model=AgentListResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def list_agents(
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentListResponse:
    """list all AI agents."""
    try:
        agent_names = agent_service.list_agents()
        agents = []

        for name in agent_names:
            agent = agent_service.get_agent(name)
            if agent:
                agents.append(
                    AgentResponse(
                        id=agent.id,
                        name=agent.config.name,
                        provider=agent.config.provider.value,
                        model=agent.config.model,
                        status=agent.status.value,
                        usage_count=agent.usage_count,
                        created_at=agent.created_at,
                        last_used=agent.last_used,
                        tools=agent.config.tools,
                    )
                )

        return AgentListResponse(
            agents=agents,
            total=len(agents),
        )

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/{agent_name}",
    response_model=AgentResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_agent(
    agent_name: str,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    """Get an AI agent by name."""
    try:
        agent = agent_service.get_agent(agent_name)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found",
            )

        return AgentResponse(
            id=agent.id,
            name=agent.config.name,
            provider=agent.config.provider.value,
            model=agent.config.model,
            status=agent.status.value,
            usage_count=agent.usage_count,
            created_at=agent.created_at,
            last_used=agent.last_used,
            tools=agent.config.tools,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.put(
    "/{agent_name}",
    response_model=AgentResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def update_agent(
    agent_name: str,
    request: UpdateAgentRequest,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    """Update an AI agent."""
    try:
        # Get current agent
        current_agent = agent_service.get_agent(agent_name)
        if not current_agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found",
            )

        # Update config with provided values
        config = current_agent.config.model_copy()
        if request.provider is not None:
            config.provider = request.provider
        if request.model is not None:
            config.model = request.model
        if request.system_prompt is not None:
            config.system_prompt = request.system_prompt
        if request.temperature is not None:
            config.temperature = request.temperature
        if request.max_tokens is not None:
            config.max_tokens = request.max_tokens
        if request.timeout is not None:
            config.timeout = request.timeout
        if request.max_retries is not None:
            config.max_retries = request.max_retries
        if request.tools is not None:
            config.tools = request.tools

        # Update agent
        success = agent_service.update_agent_config(agent_name, config)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update agent",
            )

        # Return updated agent
        updated_agent = agent_service.get_agent(agent_name)
        if not updated_agent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve updated agent",
            )

        return AgentResponse(
            id=updated_agent.id,
            name=updated_agent.config.name,
            provider=updated_agent.config.provider.value,
            model=updated_agent.config.model,
            status=updated_agent.status.value,
            usage_count=updated_agent.usage_count,
            created_at=updated_agent.created_at,
            last_used=updated_agent.last_used,
            tools=updated_agent.config.tools,
        )

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete(
    "/{agent_name}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def delete_agent(
    agent_name: str,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Delete an AI agent."""
    try:
        success = agent_service.delete_agent(agent_name)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/{agent_name}/stats",
    response_model=AgentStatsResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_agent_stats(
    agent_name: str,
    agent_service: AgentService = Depends(get_agent_service),
) -> AgentStatsResponse:
    """Get agent statistics."""
    try:
        stats = agent_service.get_agent_stats(agent_name)
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent '{agent_name}' not found",
            )

        return AgentStatsResponse(**stats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/system/stats",
    response_model=SystemStatsResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def get_system_stats(
    agent_service: AgentService = Depends(get_agent_service),
) -> SystemStatsResponse:
    """Get system statistics."""
    try:
        stats = agent_service.get_all_stats()
        return SystemStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/conversations/{session_id}",
    response_model=ConversationHistoryResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def get_conversation_history(
    session_id: str,
    agent_service: AgentService = Depends(get_agent_service),
) -> ConversationHistoryResponse:
    """Get conversation history for a session."""
    try:
        conversation = agent_service.get_conversation_history(session_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{session_id}' not found",
            )

        return ConversationHistoryResponse(
            session_id=conversation.session_id,
            agent_name=conversation.agent_name,
            messages=[msg.model_dump() for msg in conversation.messages],
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.delete(
    "/conversations/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def clear_conversation_history(
    session_id: str,
    agent_service: AgentService = Depends(get_agent_service),
):
    """Clear conversation history for a session."""
    try:
        success = agent_service.clear_conversation_history(session_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation '{session_id}' not found",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/tools/",
    response_model=ToolListResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def list_tools(
    agent_service: AgentService = Depends(get_agent_service),
) -> ToolListResponse:
    """list all available tools."""
    try:
        tool_names = agent_service.get_available_tools()
        tools = []

        for name in tool_names:
            definition = agent_service.tool_registry.get_tool_definition(name)
            if definition:
                tools.append(
                    ToolDefinitionResponse(
                        name=definition.name,
                        description=definition.description,
                        parameters=definition.parameters,
                        required=definition.required,
                    )
                )

        return ToolListResponse(
            tools=tools,
            total=len(tools),
        )

    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        500: {"model": ErrorResponse},
    },
)
async def health_check(
    agent_service: AgentService = Depends(get_agent_service),
) -> HealthResponse:
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            agents_count=len(agent_service.list_agents()),
        )

    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
