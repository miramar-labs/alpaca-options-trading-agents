from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Signal(BaseModel):
    """Structured output of the Dealer's LLM call — replaces gpt-trader.py's xtractjson() hack.
    The Dealer's graph branches on `action` via a deterministic edge; the LLM never calls
    Floor Broker directly."""

    symbol: str
    action: Literal["BUY", "HOLD", "SELL"]
    reasoning: str = Field(description="Explanation citing the indicators and news feed that led to this decision")
    size_hint: float = Field(default=1.0, ge=0.0, le=1.0, description="Fraction of the symbol's budget to deploy on a BUY")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How strongly the indicators agree with and support this action, from 0.0 (weak or "
            "mixed signal) to 1.0 (multiple indicators clearly agree). A borderline or conflicting "
            "reading should score low, not 1.0."
        ),
    )


class OptionContractPick(BaseModel):
    """Structured output of the Dealer's MCP-backed option contract search -- the LLM has already
    called Alpaca MCP tools (chain search, quotes, Greeks) before producing this, so every field
    here reflects a real contract it found, not a guess."""

    contract_symbol: str
    strike: float
    expiration: str = Field(description="ISO date (YYYY-MM-DD) of the contract's expiration")
    right: Literal["call", "put"]
    delta: float
    premium: float = Field(gt=0, description="Mid-price premium per share observed via the MCP quote tool")
    reasoning: str = Field(description="Why this specific contract was chosen over other candidates in the chain")

    @field_validator("expiration")
    @classmethod
    def _expiration_is_valid_iso_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"expiration must be an ISO date (YYYY-MM-DD), got {v!r}") from exc
        return v
