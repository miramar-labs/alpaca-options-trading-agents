from typing import List

from pydantic import BaseModel, Field


class CandidateResearch(BaseModel):
    symbol: str
    # Not constrained to a Literal: the allowed value ("stocks" or cfg.trading.crypto_taapi_exchange)
    # is config-driven, not fixed. graph.py's validate_selection() overrides this against the actual
    # candidate's known market rather than trusting the LLM's copy of the field.
    exchange: str = "stocks"
    budget: float
    indicators: List[str]
    rationale: str = Field(description="One-line reason this candidate made the book")


class PortfolioSelection(BaseModel):
    """Structured output of the Analyst's LLM selection node — written to the `portfolio` ConfigMap."""

    symbols: List[CandidateResearch]
