"""
Tool system for agents with built-in tools and custom tool registration.
"""

import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

try:
    from pydantic_ai import RunContext
except ImportError:
    RunContext = Any

from .models import AgentDependencies, ToolDefinition

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._tool_definitions: dict[str, ToolDefinition] = {}
        self._register_builtin_tools()

    def register_tool(
        self,
        name: str,
        func: Callable,
        description: str,
        parameters: Optional[dict[str, Any]] = None,
        required: Optional[list[str]] = None,
    ) -> None:
        """Register a tool with the registry."""
        self._tools[name] = func
        self._tool_definitions[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            required=required or [],
        )
        logger.info(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tool_definition(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool definition by name."""
        return self._tool_definitions.get(name)

    def list_tools(self) -> list[str]:
        """list all registered tool names."""
        return list(self._tools.keys())

    def get_all_tools(self) -> dict[str, Callable]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_all_definitions(self) -> dict[str, ToolDefinition]:
        """Get all tool definitions."""
        return self._tool_definitions.copy()

    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Time and date tools
        self.register_tool(
            name="get_current_time",
            func=get_current_time,
            description="Get the current date and time",
            parameters={},
            required=[],
        )

        # File system tools
        self.register_tool(
            name="read_file",
            func=read_file,
            description="Read the contents of a file",
            parameters={"file_path": {"type": "string", "description": "Path to the file to read"}},
            required=["file_path"],
        )

        self.register_tool(
            name="write_file",
            func=write_file,
            description="Write content to a file",
            parameters={
                "file_path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            required=["file_path", "content"],
        )

        self.register_tool(
            name="list_directory",
            func=list_directory,
            description="list contents of a directory",
            parameters={"directory_path": {"type": "string", "description": "Path to the directory to list"}},
            required=["directory_path"],
        )

        # System tools
        self.register_tool(
            name="run_command",
            func=run_command,
            description="Execute a system command",
            parameters={
                "command": {"type": "string", "description": "Command to execute"},
                "timeout": {"type": "number", "description": "Timeout in seconds (default: 30)"},
            },
            required=["command"],
        )

        # Web tools
        self.register_tool(
            name="make_http_request",
            func=make_http_request,
            description="Make an HTTP request",
            parameters={
                "url": {"type": "string", "description": "URL to request"},
                "method": {"type": "string", "description": "HTTP method (GET, POST, etc.)"},
                "headers": {"type": "object", "description": "HTTP headers"},
                "data": {"type": "object", "description": "Request data"},
            },
            required=["url"],
        )

        # Math tools
        self.register_tool(
            name="calculate",
            func=calculate,
            description="Perform mathematical calculations",
            parameters={"expression": {"type": "string", "description": "Mathematical expression to evaluate"}},
            required=["expression"],
        )


# Global tool registry instance
tool_registry = ToolRegistry()


# Built-in tool functions
async def get_current_time(ctx: RunContext[AgentDependencies]) -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


async def read_file(ctx: RunContext[AgentDependencies], file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except PermissionError:
        return f"Error: Permission denied to read '{file_path}'"
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def write_file(ctx: RunContext[AgentDependencies], file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'"
    except PermissionError:
        return f"Error: Permission denied to write to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"


async def list_directory(ctx: RunContext[AgentDependencies], directory_path: str) -> str:
    """list contents of a directory."""
    try:
        items = os.listdir(directory_path)
        return json.dumps(items, indent=2)
    except FileNotFoundError:
        return f"Error: Directory '{directory_path}' not found"
    except PermissionError:
        return f"Error: Permission denied to access '{directory_path}'"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


async def run_command(ctx: RunContext[AgentDependencies], command: str, timeout: int = 30) -> str:
    """Execute a system command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

        return json.dumps(output, indent=2)
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


async def make_http_request(
    ctx: RunContext[AgentDependencies],
    url: str,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    data: Optional[dict[str, Any]] = None,
) -> str:
    """Make an HTTP request."""
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=data,
            )

            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text,
            }

            return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error making HTTP request: {str(e)}"


async def calculate(ctx: RunContext[AgentDependencies], expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # Simple evaluation - in production, use a safer math parser
        import math

        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "pi": math.pi,
            "e": math.e,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
        }

        # Basic safety check
        if any(dangerous in expression for dangerous in ["import", "exec", "eval", "__"]):
            return "Error: Unsafe expression detected"

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating expression: {str(e)}"


def create_tool_decorator(registry: ToolRegistry):
    """Create a decorator for registering tools."""

    def tool(
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[dict[str, Any]] = None,
        required: Optional[list[str]] = None,
    ):
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or f"Tool: {tool_name}"

            registry.register_tool(
                name=tool_name,
                func=func,
                description=tool_description,
                parameters=parameters,
                required=required,
            )
            return func

        return decorator

    return tool


# Create the tool decorator
tool = create_tool_decorator(tool_registry)


# Example of custom tool registration
@tool(
    name="example_tool",
    description="An example custom tool",
    parameters={"input_text": {"type": "string", "description": "Input text to process"}},
    required=["input_text"],
)
async def example_tool(ctx: RunContext[AgentDependencies], input_text: str) -> str:
    """Example custom tool."""
    return f"Processed: {input_text}"
