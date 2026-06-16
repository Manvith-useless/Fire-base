"""Technical & Quant module.

Interprets the deterministically-computed indicator snapshot and emits scores for
three dimensions: trend_momentum, liquidity, and catalysts.
"""
from __future__ import annotations

from ..context import AnalysisContext
from ..llm import LLM
from ..schemas import AgentReport
from .base import Agent, fmt


class TechnicalQuantAgent(Agent):
    name = "Technical & Quant"

    def run(self, ctx: AnalysisContext, llm: LLM) -> list[AgentReport]:
        t = ctx.technical
        close = t.get("close")

        # ---- Trend & momentum -------------------------------------------------
        trend_score = 50.0
        ev: list[str] = []
        risks: list[str] = []

        sma20, sma50, sma200 = t.get("sma_20"), t.get("sma_50"), t.get("sma_200")
        if close and sma50 and sma200:
            if close > sma50 > sma200:
                trend_score += 18
                ev.append(f"Price {fmt(close)} > SMA50 {fmt(sma50)} > SMA200 {fmt(sma200)} (bullish stack)")
            elif close < sma50 < sma200:
                trend_score -= 18
                risks.append(f"Price {fmt(close)} < SMA50 < SMA200 (bearish stack)")
            else:
                ev.append("Moving averages mixed (no clean trend stack)")

        rsi = t.get("rsi_14")
        if rsi is not None:
            if rsi > 70:
                trend_score -= 6
                risks.append(f"RSI {fmt(rsi)} overbought")
            elif rsi < 30:
                trend_score += 6
                ev.append(f"RSI {fmt(rsi)} oversold (possible mean-reversion upside)")
            else:
                ev.append(f"RSI {fmt(rsi)} neutral")
                if rsi > 50:
                    trend_score += 5

        macd_hist = t.get("macd_hist")
        if macd_hist is not None:
            if macd_hist > 0:
                trend_score += 8
                ev.append(f"MACD histogram positive ({fmt(macd_hist, 3)}) — upward momentum")
            else:
                trend_score -= 8
                risks.append(f"MACD histogram negative ({fmt(macd_hist, 3)}) — fading momentum")

        adx = t.get("adx_14")
        plus_di, minus_di = t.get("plus_di"), t.get("minus_di")
        if adx is not None:
            if adx > 25 and plus_di and minus_di:
                if plus_di > minus_di:
                    trend_score += 6
                    ev.append(f"ADX {fmt(adx)} strong trend, +DI>-DI (bullish)")
                else:
                    trend_score -= 6
                    risks.append(f"ADX {fmt(adx)} strong trend, -DI>+DI (bearish)")
            else:
                ev.append(f"ADX {fmt(adx)} — weak/no trend")

        trend = self._report("trend_momentum", trend_score,
                             confidence=78 if close else 40,
                             evidence=ev or ["Insufficient trend signal"],
                             risks=risks or ["None material"])

        # ---- Liquidity --------------------------------------------------------
        liq_score = 55.0
        liq_ev: list[str] = []
        liq_risks: list[str] = []
        vol = ctx.candles["volume"] if "volume" in ctx.candles else None
        if vol is not None and len(vol) >= 20:
            recent = vol.tail(20).mean()
            longer = vol.tail(100).mean() if len(vol) >= 100 else recent
            liq_ev.append(f"Avg volume (20d): {recent:,.0f}")
            if recent >= longer:
                liq_score += 12
                liq_ev.append("Recent volume >= longer-term average (healthy participation)")
            else:
                liq_score -= 8
                liq_risks.append("Recent volume below longer-term average (thinning interest)")
            if recent < 50_000:
                liq_score -= 20
                liq_risks.append("Low absolute volume — slippage/exit risk")
        liquidity = self._report("liquidity", liq_score, confidence=70,
                                 evidence=liq_ev or ["No volume data"],
                                 risks=liq_risks or ["None material"])

        # ---- Catalysts (price-action) ----------------------------------------
        cat_score = 50.0
        cat_ev: list[str] = []
        cat_risks: list[str] = []
        bb_upper, bb_lower = t.get("bb_upper"), t.get("bb_lower")
        if close and bb_upper and bb_lower:
            if close > bb_upper:
                cat_score += 12
                cat_ev.append(f"Price {fmt(close)} broke above upper Bollinger {fmt(bb_upper)} (breakout)")
            elif close < bb_lower:
                cat_score -= 10
                cat_risks.append(f"Price {fmt(close)} below lower Bollinger {fmt(bb_lower)} (breakdown)")
            else:
                cat_ev.append("Price within Bollinger bands (no breakout catalyst)")
        # Proximity to 52-week (approx from available window) high.
        if len(ctx.candles) >= 50 and close:
            window_high = float(ctx.candles["high"].tail(250).max())
            if close >= 0.98 * window_high:
                cat_score += 8
                cat_ev.append("Trading near window high (momentum catalyst)")
        catalysts = self._report("catalysts", cat_score, confidence=60,
                                 evidence=cat_ev or ["No clear catalyst"],
                                 risks=cat_risks or ["None material"])

        return [trend, liquidity, catalysts]
