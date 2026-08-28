import os

# src/common/alpaca_client.py builds a TradingClient at import time -- it doesn't make any
# network calls until a request is issued, but it does require non-empty credentials to
# construct, so tests need dummy values present before anything under src/ is imported.
os.environ.setdefault("ALPACA_PAPER_API_KEY", "test")
os.environ.setdefault("ALPACA_PAPER_API_SECRET", "test")

# src/common/slack.py reads SLACK_WEBHOOK_URL2 at import time and posts to it fire-and-forget
# on any error/notification path, regardless of what a test is actually exercising. Unlike the
# Alpaca credentials above, this must override (not setdefault) whatever is in the ambient shell
# environment -- a real webhook exported there (e.g. for local dev against the live cluster)
# would otherwise let an unmocked exception path in a test post a real Slack message. See the
# incident this fixed: a manual "revert the fix and confirm the regression test catches it" run
# against tests/floor_broker/test_app.py posted a real error for its "MGN" fixture symbol.
os.environ["SLACK_WEBHOOK_URL2"] = ""
