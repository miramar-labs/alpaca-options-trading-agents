from alpaca.trading.enums import AssetClass

from src.common.alpaca_client import trading_client


def fetch_fills(date: str, only_crypto: bool | None = None) -> list[dict]:
    """Fetches an account's FILL activities for one date. `only_crypto=True` keeps only crypto
    fills, `False` keeps only equity fills, `None` (default) keeps both -- matching the "/" in
    symbol convention already used for crypto elsewhere (e.g. alpaca_client.get_current_ask_price)
    since Alpaca's activities payload carries no separate asset-class field. Confirmed against a
    live account: fill/activity symbols do carry the slash for crypto (e.g. "BTC/USD")."""
    raw = trading_client.get(
        "/account/activities",
        data={"activity_types": "FILL", "date": date},
    )
    fills = [
        {
            "symbol": f["symbol"],
            "side": f["side"],
            "qty": float(f["qty"]),
            "price": float(f["price"]),
            "time": f["transaction_time"],
        }
        for f in raw
    ]
    if only_crypto is None:
        return fills
    return [f for f in fills if ("/" in f["symbol"]) == only_crypto]


def summarize_positions(positions, only_crypto: bool | None = None) -> list[dict]:
    """Shapes Alpaca Position objects into the plain dicts slack.notify_*_report() and the
    Analyst's fetch_position_pnl() expect. Unlike fetch_fills(), this filters on `p.asset_class`,
    not a "/" in the symbol -- confirmed against a live account that Alpaca's Position.symbol for
    crypto has no slash (e.g. "BTCUSD"), unlike the activities/fills payload which does (e.g.
    "BTC/USD"). Matches the asset_class check portfolio_state.merge_held_positions() already uses
    for the same reason. `unrealized_pl` and `current_price` are Optional[str] on Alpaca's own
    Position model, so they're guarded and may come back None here too -- callers must handle that."""
    if only_crypto is not None:
        positions = [p for p in positions if (p.asset_class == AssetClass.CRYPTO) == only_crypto]
    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "unrealized_plpc": float(p.unrealized_plpc),
            "unrealized_pl": float(p.unrealized_pl) if p.unrealized_pl is not None else None,
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price is not None else None,
        }
        for p in positions
    ]
