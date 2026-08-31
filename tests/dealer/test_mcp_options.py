import asyncio

from src.dealer import mcp_options


def test_get_options_tools_config_uses_read_only_toolsets(monkeypatch):
    mcp_options.reset_options_tools_cache()
    captured = {}

    class FakeClient:
        def __init__(self, connections):
            captured["connections"] = connections

        async def get_tools(self):
            return ["fake-tool"]

    monkeypatch.setenv("ALPACA_ALT_KEY", "test-key")
    monkeypatch.setenv("ALPACA_ALT_SECRET", "test-secret")
    monkeypatch.setattr(mcp_options, "live_account_env_names", lambda: ("ALPACA_ALT_KEY", "ALPACA_ALT_SECRET"))
    monkeypatch.setattr(mcp_options, "MultiServerMCPClient", FakeClient)

    tools = asyncio.run(mcp_options.get_options_tools())

    assert tools == ["fake-tool"]
    conn = captured["connections"]["alpaca"]
    assert conn["transport"] == "stdio"
    assert conn["command"] == "alpaca-mcp-server"
    assert conn["env"]["ALPACA_API_KEY"] == "test-key"
    assert conn["env"]["ALPACA_SECRET_KEY"] == "test-secret"
    assert conn["env"]["ALPACA_TOOLSETS"] == "assets,options-data,account"


def test_get_options_tools_is_cached_until_reset(monkeypatch):
    mcp_options.reset_options_tools_cache()
    builds = {"count": 0}

    class FakeClient:
        def __init__(self, connections):
            builds["count"] += 1

        async def get_tools(self):
            return ["fake-tool"]

    monkeypatch.setenv("ALPACA_ALT_KEY", "test-key")
    monkeypatch.setenv("ALPACA_ALT_SECRET", "test-secret")
    monkeypatch.setattr(mcp_options, "live_account_env_names", lambda: ("ALPACA_ALT_KEY", "ALPACA_ALT_SECRET"))
    monkeypatch.setattr(mcp_options, "MultiServerMCPClient", FakeClient)

    asyncio.run(mcp_options.get_options_tools())
    asyncio.run(mcp_options.get_options_tools())
    assert builds["count"] == 1  # second call served from cache

    mcp_options.reset_options_tools_cache()
    asyncio.run(mcp_options.get_options_tools())
    assert builds["count"] == 2  # reset forces a rebuild
