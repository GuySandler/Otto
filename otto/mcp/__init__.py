"""MCP (Model Context Protocol) support: client, tool policies, per-user manager."""

from .client import MCPCallResult, MCPClient, MCPError, MCPToolInfo
from .manager import MCPManager, ServerState
from .policy import POLICIES, ToolPolicy, policy_for

__all__ = [
    "MCPCallResult",
    "MCPClient",
    "MCPError",
    "MCPManager",
    "MCPToolInfo",
    "POLICIES",
    "ServerState",
    "ToolPolicy",
    "policy_for",
]
