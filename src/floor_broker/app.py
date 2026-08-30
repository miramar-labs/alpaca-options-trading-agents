import re
from datetime import datetime
from typing import Literal

from alpaca.common.exceptions import APIError
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.common import slack
from src.common.logging import get_logger
from src.floor_broker import execution

log = get_logger("FLOOR")

# Sanity ceiling on a single order's authorized budget (ROADMAP P0.3) -- not a business rule,
# just a last-line-of-defense guard against a units/config bug or a hallucinated Analyst budget
# (analyst/schema.py's `budget` field has no upper bound of its own) reaching Alpaca as a live
# order. 20x config.yaml's analyst.default_budget (5000).
MAX_BUDGET = 100_000.0

# Same category of guard as MAX_BUDGET above, for the options path -- a hard, non-configurable
# ceiling on a single option order's notional (qty * premium * 100) as a last-line-of-defense
# against a units bug or a wildly hallucinated premium/qty reaching Alpaca. Distinct from
# config.yaml's options_trading.max_notional_usd, which is the real, configurable, re-quoted-
# against-the-market business-rule cap enforced inside buy_option() itself (Step 3 below) --
# this pydantic ceiling is only the outer sanity bound on the raw request.
MAX_OPTION_NOTIONAL = 100_000.0

# Alpaca tickers: letters/digits, with an optional single "/" for crypto pairs (e.g. "BTC/USD")
# or "." for dual-class shares and warrants/units (e.g. "BRK.B", "DSX.WS") -- both come straight
# from Alpaca's own screener/assets universe, so Floor Broker must accept whatever shape Alpaca
# itself vends rather than second-guessing it; a genuinely bad symbol still gets a clean rejection
# from Alpaca's own API (caught as an APIError) instead of a client-side ValueError here.
_SYMBOL_RE = re.compile(r"^[A-Z0-9]{1,10}([./][A-Z0-9]{1,10})?$")
# "stocks", or a TAAPI venue identifier (e.g. "binance") -- config-driven (cfg.trading.
# crypto_taapi_exchange), so this validates shape, not a fixed enum of known venues.
_EXCHANGE_RE = re.compile(r"^[a-z0-9_-]+$")


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    exchange: str
    action: Literal["BUY", "SELL"]
    # execution.sell() derives qty to sell from the live open position, not from budget -- a
    # held-only position (merge_held_positions()) legitimately carries budget=0.0, so budget is
    # only a real business rule (authorized new-BUY capital) for BUY. See _budget_required_for_buy.
    budget: float = Field(ge=0, le=MAX_BUDGET)
    slP: float = Field(gt=0, lt=1)
    tpP: float = Field(gt=1, lt=2)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        v = v.strip().upper()
        if not _SYMBOL_RE.match(v):
            raise ValueError(f"invalid symbol: {v!r}")
        return v

    @field_validator("budget")
    @classmethod
    def _budget_required_for_buy(cls, v: float, info) -> float:
        if info.data.get("action") == "BUY" and v <= 0:
            raise ValueError("budget must be greater than 0 for BUY")
        return v

    @field_validator("exchange")
    @classmethod
    def _normalize_exchange(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EXCHANGE_RE.match(v):
            raise ValueError(f"invalid exchange: {v!r}")
        return v


class ExecuteResponse(BaseModel):
    status: Literal["executed", "submitted", "skipped", "error"]
    detail: str
    reason: str | None = None
    order_id: str | None = None
    fill_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None


class ExecuteOptionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_symbol: str
    side: Literal["BUY"]
    qty: int = Field(gt=0)
    symbol: str
    right: Literal["call", "put"]
    strike: float = Field(gt=0)
    expiration: str
    delta: float | None = None
    premium: float = Field(gt=0)
    reasoning: str | None = None
    cycle_id: str | None = None

    @field_validator("premium")
    @classmethod
    def _notional_within_ceiling(cls, v: float, info) -> float:
        qty = info.data.get("qty")
        if qty is not None and qty * v * 100 > MAX_OPTION_NOTIONAL:
            raise ValueError(f"notional (qty * premium * 100) exceeds ceiling of {MAX_OPTION_NOTIONAL}")
        return v

    @field_validator("expiration")
    @classmethod
    def _expiration_is_valid_iso_date(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"expiration must be an ISO date (YYYY-MM-DD), got {v!r}") from exc
        return v


class ExecuteOptionResponse(BaseModel):
    status: Literal["executed", "submitted", "skipped", "error"]
    detail: str
    reason: str | None = None
    order_id: str | None = None


class OptionExposureResponse(BaseModel):
    contracts: list[str] = Field(default_factory=list)


class FlattenCryptoResponse(BaseModel):
    status: Literal["ok", "error"]
    events: list[dict] = Field(default_factory=list)
    detail: str | None = None


class FlattenOptionsResponse(BaseModel):
    status: Literal["ok", "error"]
    events: list[dict] = Field(default_factory=list)
    detail: str | None = None


app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest):
    try:
        if req.action == "BUY":
            result = execution.buy(req.symbol, req.exchange, req.budget, req.slP, req.tpP)
        else:
            result = execution.sell(req.symbol)
        return ExecuteResponse(**result)
    except APIError as exc:
        log(f"💥  {req.action} {req.symbol} failed: {exc}")
        return ExecuteResponse(status="error", detail=str(exc))
    except Exception as exc:
        log(f"💥  unexpected error on {req.action} {req.symbol}: {exc}")
        slack.notify_error("FLOOR", f"unexpected error on {req.action} {req.symbol}: {exc}")
        raise


@app.post("/execute-option", response_model=ExecuteOptionResponse)
def execute_option(req: ExecuteOptionRequest):
    try:
        result = execution.buy_option(
            req.contract_symbol,
            req.qty,
            req.premium,
            req.right,
            req.strike,
            req.expiration,
            req.delta,
            req.reasoning,
            req.symbol,
            req.cycle_id,
        )
        return ExecuteOptionResponse(**result)
    except APIError as exc:
        log(f"💥  option BUY {req.contract_symbol} failed: {exc}")
        return ExecuteOptionResponse(status="error", detail=str(exc))
    except Exception as exc:
        log(f"💥  unexpected error on option BUY {req.contract_symbol}: {exc}")
        slack.notify_error("FLOOR", f"unexpected error on option BUY {req.contract_symbol}: {exc}")
        raise


@app.get("/option-exposure", response_model=OptionExposureResponse)
def option_exposure():
    """Contracts already held or with a BUY order in flight. The Dealer checks this before a new
    option entry to skip a duplicate before spending a Slack line and an /execute-option round trip;
    buy_option() enforces the same rule server-side regardless."""
    return OptionExposureResponse(contracts=execution.option_exposure_contract_symbols())


@app.post("/flatten-crypto", response_model=FlattenCryptoResponse)
def flatten_crypto():
    """Called by power_scheduler right before it scales this pod to 0 -- force-sells every open
    crypto position since crypto's stop-loss/take-profit is only enforced by this process's own
    poll loop (no Alpaca server-side bracket support for crypto)."""
    try:
        events = execution.flatten_all_crypto()
        return FlattenCryptoResponse(status="ok", events=events)
    except APIError as exc:
        log(f"💥  flatten-crypto failed: {exc}")
        return FlattenCryptoResponse(status="error", detail=str(exc))
    except Exception as exc:
        log(f"💥  unexpected error on flatten-crypto: {exc}")
        slack.notify_error("FLOOR", f"unexpected error on flatten-crypto: {exc}")
        raise


@app.post("/flatten-options", response_model=FlattenOptionsResponse)
def flatten_options():
    """Called by power_scheduler right before it scales this pod to 0 -- force-sells every open
    option position since check_option_stops()'s SL/TP/DTE-force-close is only enforced by this
    process's own poll loop (no Alpaca server-side bracket support for options either)."""
    try:
        events = execution.flatten_all_options()
        return FlattenOptionsResponse(status="ok", events=events)
    except APIError as exc:
        log(f"💥  flatten-options failed: {exc}")
        return FlattenOptionsResponse(status="error", detail=str(exc))
    except Exception as exc:
        log(f"💥  unexpected error on flatten-options: {exc}")
        slack.notify_error("FLOOR", f"unexpected error on flatten-options: {exc}")
        raise
