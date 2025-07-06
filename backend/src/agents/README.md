# Agent-Based AI System

A comprehensive agent-based AI system built with PydanticAI, providing provider-agnostic AI interactions, tool support, and structured responses.

## Features

- **Provider-Agnostic**: Support for multiple AI providers (OpenAI, Anthropic, Google, Groq, Cohere, Mistral)
- **Tool System**: Built-in tools and custom tool registration
- **Structured Responses**: Pydantic models for type-safe interactions
- **Conversation History**: Session-based conversation management
- **Streaming Support**: Real-time response streaming
- **Dependency Injection**: FastAPI-style dependency injection
- **Agent Management**: Create, update, delete, and monitor agents
- **Statistics & Monitoring**: Comprehensive agent and system statistics

## Architecture

### Core Components

1. **Models** (`models.py`): Pydantic models for data structures
2. **AI Client** (`ai_client.py`): Provider-agnostic AI client wrapper
3. **Tools** (`tools.py`): Tool registry and built-in tools
4. **Services** (`services.py`): Agent service for orchestration
5. **Router** (`router.py`): FastAPI endpoints
6. **Dependencies** (`dependencies.py`): Dependency injection
7. **Schemas** (`schemas.py`): API request/response schemas

### Built-in Tools

- **Time Tools**: `get_current_time`
- **File System Tools**: `read_file`, `write_file`, `list_directory`
- **System Tools**: `run_command`
- **Web Tools**: `make_http_request`
- **Math Tools**: `calculate`

## Usage

### Basic Agent Usage

```python
from agents import AgentService, AgentConfig, AIProvider

# Create agent service
service = AgentService()

# Create a custom agent
config = AgentConfig(
    name="my_agent",
    provider=AIProvider.OPENAI,
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    tools=["get_current_time", "calculate"]
)

agent = service.create_agent(config)

# Chat with agent
response = await service.chat_with_agent(
    agent_name="my_agent",
    message="What time is it?",
    session_id="user_123"
)

print(response.content)
```

### Custom Tool Registration

```python
from agents import tool_registry, tool

@tool(
    name="custom_tool",
    description="A custom tool",
    parameters={
        "input": {"type": "string", "description": "Input text"}
    },
    required=["input"]
)
async def my_custom_tool(ctx, input: str) -> str:
    return f"Processed: {input}"

# Tool is automatically registered
```

### API Endpoints

The agents module provides comprehensive REST API endpoints:

#### Chat Endpoints

- `POST /agents/chat` - Chat with an agent
- `POST /agents/chat/stream` - Stream chat with an agent

#### Agent Management

- `GET /agents/` - list all agents
- `POST /agents/` - Create a new agent
- `GET /agents/{agent_name}` - Get agent details
- `PUT /agents/{agent_name}` - Update agent
- `DELETE /agents/{agent_name}` - Delete agent

#### Monitoring

- `GET /agents/{agent_name}/stats` - Get agent statistics
- `GET /agents/system/stats` - Get system statistics

#### Conversation Management

- `GET /agents/conversations/{session_id}` - Get conversation history
- `DELETE /agents/conversations/{session_id}` - Clear conversation history

#### Tools

- `GET /agents/tools/` - list available tools

#### Health Check

- `GET /agents/health` - Health check endpoint

### Example API Usage

```bash
# Create an agent
curl -X POST "http://localhost:8000/api/v1/agents/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "research_bot",
    "provider": "openai",
    "model": "gpt-4o",
    "system_prompt": "You are a research assistant.",
    "tools": ["make_http_request", "calculate"]
  }'

# Chat with agent
curl -X POST "http://localhost:8000/api/v1/agents/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather like?",
    "agent_name": "research_bot",
    "session_id": "session_123"
  }'

# Get agent stats
curl "http://localhost:8000/api/v1/agents/research_bot/stats"
```

## Configuration

### Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Google
GOOGLE_API_KEY=your_google_key
GOOGLE_MODEL=gemini-1.5-flash

# Groq
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-70b-versatile

# Cohere
COHERE_API_KEY=your_cohere_key
COHERE_MODEL=command-r-plus

# Mistral
MISTRAL_API_KEY=your_mistral_key
MISTRAL_MODEL=mistral-large-latest

# Agent Settings
DEFAULT_AI_PROVIDER=openai
AGENT_MAX_RETRIES=3
AGENT_TIMEOUT=30
AGENT_TEMPERATURE=0.7
```

## Default Agents

The system creates three default agents:

1. **general_assistant**: General-purpose assistant with basic tools
2. **code_assistant**: Specialized for coding tasks with file system tools
3. **research_assistant**: Optimized for research with web tools

## Tool Development

### Creating Custom Tools

```python
from agents import tool_registry
from pydantic_ai import RunContext
from agents.models import AgentDependencies

async def my_tool(ctx: RunContext[AgentDependencies], param: str) -> str:
    """Custom tool function."""
    # Access dependencies
    user_id = ctx.deps.user_id
    session_id = ctx.deps.session_id
    
    # Tool logic here
    return f"Result for {param}"

# Register manually
tool_registry.register_tool(
    name="my_tool",
    func=my_tool,
    description="My custom tool",
    parameters={"param": {"type": "string", "description": "Parameter"}},
    required=["param"]
)
```

### Tool Security

- File system tools have path restrictions
- Command execution has timeout limits
- Math evaluation uses safe evaluation
- HTTP requests have configurable timeouts

## Error Handling

The system provides comprehensive error handling:

- **Validation Errors**: Pydantic validation for all inputs
- **Agent Errors**: Proper error propagation from AI providers
- **Tool Errors**: Safe error handling in tool execution
- **HTTP Errors**: Proper HTTP status codes and error responses

## Monitoring & Statistics

### Agent Statistics

- Usage count
- Last used timestamp
- Status (idle, processing, error, disabled)
- Configuration details

### System Statistics

- Total agents
- Total conversations
- Available tools
- Agent performance metrics

## Future Enhancements

- **Vector Database Integration**: For embeddings and semantic search
- **Memory Management**: Persistent conversation memory
- **Agent Workflows**: Multi-step agent workflows
- **Plugin System**: Dynamic tool loading
- **Performance Monitoring**: Detailed performance metrics
- **Agent Clustering**: Distributed agent management

## Security Considerations

- API key management through environment variables
- Tool execution sandboxing
- Input validation and sanitization
- Rate limiting (to be implemented)
- Authentication integration with existing auth system

## Development

### Running Tests

```bash
# Run agent tests
pytest backend/tests/agents/

# Run with coverage
pytest --cov=src/agents backend/tests/agents/
```

### Code Quality

```bash
# Format code
ruff format backend/src/agents/

# Lint code
ruff check backend/src/agents/

# Type checking
mypy backend/src/agents/
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure type safety with Pydantic models
5. Follow security best practices for tool development
