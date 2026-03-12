import os
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd
import requests
import yfinance as yf


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PRICE_HISTORY_PERIOD = "1y"
RET_3M_LOOKBACK = 63


# -----------------------------
# 리더 바스켓 (11섹터)
# 티커 / 한글명 / 섹터
# -----------------------------
LEADERS: Dict[str, Dict] = {
    "MSFT": {"name": "마이크로소프트", "sector": "기술"},
    "NVDA": {"name": "엔비디아", "sector": "반도체"},
    "GOOG": {"name": "알파벳", "sector": "통신"},
    "AMZN": {"name": "아마존", "sector": "소비재"},
    "COST": {"name": "코스트코", "sector": "필수소비"},
    "JPM": {"name": "JP모건", "sector": "금융"},
    "CAT": {"name": "캐터필러", "sector": "산업재"},
    "XOM": {"name": "엑슨모빌", "sector": "에너지"},
    "LLY": {"name": "일라이릴리", "sector": "헬스케어"},
    "NEE": {"name": "넥스트에라 에너지", "sector": "유틸리티"},
    "LIN": {"name": "린데", "sector": "소재"},
}


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    requests.post(url, json=payload, timeout=30)


def safe_float(x) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def pct(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v*100:.1f}%"


def short_number(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    return f"{x:.2f}"


def download_series(ticker: str):
    df = yf.download(
        ticker,
        period=PRICE_HISTORY_PERIOD,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise RuntimeError(f"{ticker} 데이터 없음")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()

    return close


def ma(series: pd.Series, n: int):
    if len(series) < n:
        return None
    return safe_float(series.rolling(n).mean().iloc[-1])


def ret(series: pd.Series, lookback: int):
    if len(series) <= lookback:
        return None
    start = safe_float(series.iloc[-lookback - 1])
    end = safe_float(series.iloc[-1])
    if start is None or end is None or start <= 0:
        return None
    return (end / start) - 1


def leader_analysis():

    rows = []

    above50 = 0
    above200 = 0
    retpos = 0

    for ticker, info in LEADERS.items():

        s = download_series(ticker)

        close = safe_float(s.iloc[-1])
        ma50 = ma(s, 50)
        ma200 = ma(s, 200)
        r3 = ret(s, RET_3M_LOOKBACK)

        a50 = close > ma50 if ma50 else False
        a200 = close > ma200 if ma200 else False

        if a50:
            above50 += 1

        if a200:
            above200 += 1

        if r3 and r3 > 0:
            retpos += 1

        rows.append(
            {
                "ticker": ticker,
                "name": info["name"],
                "sector": info["sector"],
                "a50": a50,
                "a200": a200,
                "ret3": r3,
            }
        )

    return rows, above50, above200, retpos


def leader_interpret(a200: int):

    if a200 >= 9:
        return "리더 구조 매우 강함"

    if a200 >= 6:
        return "리더 구조 정상"

    if a200 >= 4:
        return "리더 약화 시작"

    return "리더 붕괴"


def build_report():

    spy = download_series("SPY")
    qqq = download_series("QQQ")
    vix = download_series("^VIX")

    spy_close = safe_float(spy.iloc[-1])
    qqq_close = safe_float(qqq.iloc[-1])
    vix_close = safe_float(vix.iloc[-1])

    spy50 = ma(spy, 50)
    spy200 = ma(spy, 200)

    qqq50 = ma(qqq, 50)
    qqq200 = ma(qqq, 200)

    leaders, a50, a200, rpos = leader_analysis()

    regime = "GO"

    if spy_close < spy200 or qqq_close < qqq200:
        regime = "STOP"

    if vix_close >= 30:
        regime = "STOP"

    leader_comment = leader_interpret(a200)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = []

    lines.append(f"미국 시장 레짐 리포트 ({today})")
    lines.append("")

    lines.append(f"시장 판정: {regime}")
    lines.append("")

    lines.append("[지수 상태]")

    lines.append(
        f"SPY 50MA {'위' if spy_close>spy50 else '아래'} / 200MA {'위' if spy_close>spy200 else '아래'}"
    )

    lines.append(
        f"QQQ 50MA {'위' if qqq_close>qqq50 else '아래'} / 200MA {'위' if qqq_close>qqq200 else '아래'}"
    )

    lines.append("")
    lines.append("[VIX]")

    lines.append(f"변동성 지수: {short_number(vix_close)}")

    lines.append("")
    lines.append("[리더 상태]")

    for r in leaders:

        a50txt = "50MA 위" if r["a50"] else "50MA 아래"
        a200txt = "200MA 위" if r["a200"] else "200MA 아래"

        lines.append(
            f"{r['ticker']} {r['name']} ({r['sector']}) | {a50txt} / {a200txt} | 3M {pct(r['ret3'])}"
        )

    lines.append("")
    lines.append("[리더 구조]")

    lines.append(f"200MA 위 리더: {a200} / 11")
    lines.append(f"50MA 위 리더: {a50} / 11")
    lines.append(f"3개월 상승 리더: {rpos} / 11")

    lines.append(leader_comment)

    return "\n".join(lines)


def main():

    report = build_report()

    send_telegram(report)

    print(report)


if __name__ == "__main__":
    main()
