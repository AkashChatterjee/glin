"""Helpers that read MCP SDK results the same way on the 1.x and 2.x lines."""


async def call_structured(mcp, name, arguments):
    """mcp 1.x call_tool returns (content, structured); 2.x a CallToolResult."""
    result = await mcp.call_tool(name, arguments)
    return result[1] if isinstance(result, tuple) else result.structured_content


def input_schema(tool):
    return getattr(tool, "inputSchema", None) or tool.input_schema
