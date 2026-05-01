"""

> uv run mcp install main.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b


@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    return f"Hello {name}"


def main():
    # Initialize and run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
