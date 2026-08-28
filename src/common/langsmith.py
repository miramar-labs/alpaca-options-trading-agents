import os


def configure(cfg) -> None:
    """Wires LangGraph/LangChain tracing to LangSmith, capped by a sampling rate. Dealer's
    poll loop alone can produce thousands of traces/month across a full trading universe,
    above the free Developer plan's 5k traces/month allowance -- the sampling rate keeps
    volume under that without disabling tracing entirely."""
    if not cfg.langsmith.enabled:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith.project
    os.environ["LANGSMITH_TRACING_SAMPLING_RATE"] = str(cfg.langsmith.sampling_rate)
