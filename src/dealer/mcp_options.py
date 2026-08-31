import os

from langchain_mcp_adapters.client import MultiServerMCPClient

from src.common.alpaca_client import live_account_env_names

_TOOLS_CACHE: dict[tuple[str, str], list] = {}


def reset_options_tools_cache() -> None:
    """Called once per Dealer poll cycle. Within a cycle the Alpaca MCP tool list is stable, so
    we avoid re-spawning alpaca-mcp-server just to re-list tools for every symbol."""
    _TOOLS_CACHE.clear()


async def get_options_tools():
    """Returns LangChain-bindable tools for the official Alpaca MCP server, restricted to
    read-only toolsets -- order placement stays exclusively in Floor Broker's alpaca-py
    execution path, never via MCP tool-calling (least-privilege split, see plan Global
    Constraints). Resolves the live account's credentials the same config-driven way as
    src.common.alpaca_client's trading_client/option_data_client -- so pointing alpaca.live at a
    different paper account in config.yaml also repoints which credentials this MCP subprocess is
    launched with, not just the direct alpaca-py clients."""
    key_env, secret_env = live_account_env_names()
    cache_key = (key_env, secret_env)
    if cache_key in _TOOLS_CACHE:
        return _TOOLS_CACHE[cache_key]

    client = MultiServerMCPClient(
        {
            "alpaca": {
                "transport": "stdio",
                "command": "alpaca-mcp-server",
                "args": [],
                "env": {
                    "ALPACA_API_KEY": os.environ[key_env],
                    "ALPACA_SECRET_KEY": os.environ[secret_env],
                    "ALPACA_PAPER_TRADE": "True",
                    "ALPACA_TOOLSETS": "assets,options-data,account",
                },
            }
        }
    )
    tools = await client.get_tools()
    _TOOLS_CACHE[cache_key] = tools
    return tools
