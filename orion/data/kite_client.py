"""Live Zerodha Kite Connect data provider.

Requires a paid Kite Connect developer subscription (api_key/api_secret on
kite.trade) plus a daily access token obtained via `orion login`.

kiteconnect is imported lazily so the rest of the app (and the sample provider)
works without the package installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..config import Settings
from .provider import DataProvider, Instrument

SESSION_FILE = Path(".kite_session.json")


class KiteClient(DataProvider):
    name = "kite"

    def __init__(self, settings: Settings):
        if not settings.kite_api_key:
            raise RuntimeError("KITE_API_KEY is not set")
        try:
            from kiteconnect import KiteConnect
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise RuntimeError(
                "kiteconnect is not installed. Run: pip install kiteconnect"
            ) from e

        self.settings = settings
        self.kite = KiteConnect(api_key=settings.kite_api_key)
        access_token = settings.kite_access_token or self._cached_token()
        if not access_token:
            raise RuntimeError(
                "No Kite access token. Run `orion login` first to authenticate."
            )
        self.kite.set_access_token(access_token)
        self._instrument_cache: dict[str, int] = {}

    # ---- auth helpers -------------------------------------------------
    @staticmethod
    def _cached_token() -> str | None:
        if SESSION_FILE.exists():
            try:
                return json.loads(SESSION_FILE.read_text()).get("access_token")
            except Exception:
                return None
        return None

    @classmethod
    def login_url(cls, settings: Settings) -> str:
        from kiteconnect import KiteConnect

        return KiteConnect(api_key=settings.kite_api_key).login_url()

    @classmethod
    def complete_login(cls, settings: Settings, request_token: str) -> str:
        """Exchange request_token for an access token and cache it for the day."""
        from kiteconnect import KiteConnect

        kite = KiteConnect(api_key=settings.kite_api_key)
        data = kite.generate_session(
            request_token, api_secret=settings.kite_api_secret
        )
        token = data["access_token"]
        SESSION_FILE.write_text(json.dumps({"access_token": token}))
        os.environ["KITE_ACCESS_TOKEN"] = token
        return token

    # ---- instrument resolution ---------------------------------------
    def _instrument_token(self, instrument: Instrument) -> int:
        if instrument.key in self._instrument_cache:
            return self._instrument_cache[instrument.key]
        ltp = self.kite.ltp([instrument.key])
        token = int(ltp[instrument.key]["instrument_token"])
        self._instrument_cache[instrument.key] = token
        return token

    # ---- DataProvider API --------------------------------------------
    def get_candles(
        self, instrument: Instrument, interval: str = "day", days: int = 400
    ) -> pd.DataFrame:
        from datetime import datetime, timedelta

        token = self._instrument_token(instrument)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        records = self.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )
        return df[["date", "open", "high", "low", "close", "volume"]]

    def get_ltp(self, instrument: Instrument) -> float | None:
        data = self.kite.ltp([instrument.key])
        return float(data[instrument.key]["last_price"])

    def get_holdings(self) -> list[dict]:
        return self.kite.holdings()

    def get_margins(self) -> dict:
        return self.kite.margins()

    def place_order(self, **kwargs) -> str:
        """Place a live order. Reached ONLY through the confirmation gate."""
        return self.kite.place_order(**kwargs)
