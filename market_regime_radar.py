import os
from datetime import datetime, timezone
from typing import Optional, Dict

import pandas as pd
import requests
import yfinance as yf


# =========================================
# 환경 설정
# =========================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PRICE_HISTORY_PERIOD = "1y"
VIX_PERIOD = "6mo"

RET_3M = 63
RET_6M = 126

SP500_UNIVERSE_CSV = "data/sp500_universe.csv"


# =========================================
# 섹터 ETF
# =========================================

SECTOR_ETFS = {
    "반도체": "SMH",
    "소프트웨어": "IGV",
    "기술주": "XLK",
    "산업재": "XLI",
    "금융": "XLF",
    "헬스케어": "XLV",
    "에너지": "XLE",
    "소비재(임의소비)": "XLY",
    "필수소비재": "XLP",
    "커뮤니케이션": "XLC",
    "소재": "XLB",
    "유틸리티": "XLU",
}


# =========================================
# 섹터 대표 종목 (시장 스타일 확인용)
# =========================================

SECTOR_LEADERS: Dict[str, Dict] = {
    "MSFT": {"name": "마이크로소프트", "sector": "기술 플랫폼"},
    "NVDA": {"name": "엔비디아", "sector": "반도체"},
    "CRM": {"name": "세일즈포스", "sector": "소프트웨어"},
    "NOW": {"name": "서비스나우", "sector": "클라우드"},
    "PLTR": {"name": "팔란티어", "sector": "AI/데이터"},
    "AMZN": {"name": "아마존", "sector": "임의소비"},
    "COST": {"name": "코스트코", "sector": "필수소비"},
    "JPM": {"name": "JP모건", "sector": "금융"},
    "CAT": {"name": "캐터필러", "sector": "산업재"},
    "XOM": {"name": "엑슨모빌", "sector": "에너지"},
    "LLY": {"name": "일라이릴리", "sector": "헬스케어"},
    "NEE": {"name": "넥스트에라에너지", "sector": "유틸리티"},
    "LIN": {"name": "린데", "sector": "소재"},
}


# =========================================
# 텔레그램
# =========================================

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


# =========================================
# 유틸
# =========================================

def safe_float(x) -> Optional[float]:

    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except:
        return None


def pct(v):

    if v is None:
        return "-"

    return f"{v*100:.1f}%"


def num(v):

    if v is None:
        return "-"

    return f"{v:.2f}"


# =========================================
# 데이터 다운로드
# =========================================

def download_series(ticker, period):

    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=True,
    )

    close = df["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna()


# =========================================
# 이동평균
# =========================================

def ma(series, n):

    if len(series) < n:
        return None

    return safe_float(series.rolling(n).mean().iloc[-1])


# =========================================
# 수익률
# =========================================

def ret(series, lookback):

    if len(series) <= lookback:
        return None

    start = safe_float(series.iloc[-lookback])
    end = safe_float(series.iloc[-1])

    if start is None or end is None:
        return None

    return (end/start) - 1


# =========================================
# Breadth
# =========================================

def get_sp500():

    df = pd.read_csv(SP500_UNIVERSE_CSV)

    return df["ticker"].tolist()


def breadth():

    tickers = get_sp500()

    above50 = 0
    above200 = 0

    new_high = 0
    new_low = 0

    valid = 0

    for t in tickers:

        try:

            s = download_series(t, PRICE_HISTORY_PERIOD)

            if len(s) < 200:
                continue

            valid += 1

            last = s.iloc[-1]

            ma50 = s.rolling(50).mean().iloc[-1]
            ma200 = s.rolling(200).mean().iloc[-1]

            if last > ma50:
                above50 += 1

            if last > ma200:
                above200 += 1

            if len(s) >= 252:

                if last >= s[-252:].max():
                    new_high += 1

                if last <= s[-252:].min():
                    new_low += 1

        except:
            pass

    return {
        "above50": above50,
        "above200": above200,
        "valid": valid,
        "high": new_high,
        "low": new_low,
        "pct50": above50/valid if valid else None,
        "pct200": above200/valid if valid else None
    }


# =========================================
# 섹터 대표 종목 상태
# =========================================

def sector_leader_status():

    rows = []

    for t, info in SECTOR_LEADERS.items():

        try:

            s = download_series(t, PRICE_HISTORY_PERIOD)

            close = s.iloc[-1]

            ma50 = ma(s,50)
            ma200 = ma(s,200)

            r3 = ret(s,RET_3M)

            rows.append({
                "ticker":t,
                "name":info["name"],
                "sector":info["sector"],
                "a50": close>ma50 if ma50 else False,
                "a200": close>ma200 if ma200 else False,
                "ret3":r3
            })

        except:
            pass

    return rows


# =========================================
# 섹터 강도
# =========================================

def sector_strength():

    rows=[]

    for name,t in SECTOR_ETFS.items():

        try:

            s = download_series(t,PRICE_HISTORY_PERIOD)

            r3 = ret(s,RET_3M)
            r6 = ret(s,RET_6M)

            score = (r3 or 0)*0.45 + (r6 or 0)*0.55

            rows.append({
                "sector":name,
                "ticker":t,
                "r3":r3,
                "r6":r6,
                "score":score
            })

        except:
            pass

    rows.sort(key=lambda x:x["score"],reverse=True)

    return rows


# =========================================
# 시장 판정
# =========================================

def classify(spy_close,spy50,spy200,qqq_close,qqq50,qqq200,vix,breadth_pct):

    stop=False
    watch=False

    if breadth_pct is not None and breadth_pct < 0.30:
        stop=True

    if vix >= 30:
        stop=True

    if spy_close < spy200 or qqq_close < qqq200:
        stop=True

    if breadth_pct is not None and breadth_pct < 0.45:
        watch=True

    if vix >= 22:
        watch=True

    if spy_close < spy50 or qqq_close < qqq50:
        watch=True

    if stop:
        return "STOP"

    if watch:
        return "WATCH"

    return "GO"


# =========================================
# 리포트 생성
# =========================================

def build():

    spy = download_series("SPY",PRICE_HISTORY_PERIOD)
    qqq = download_series("QQQ",PRICE_HISTORY_PERIOD)
    vix = download_series("^VIX",VIX_PERIOD)

    spy_close = spy.iloc[-1]
    qqq_close = qqq.iloc[-1]
    vix_close = vix.iloc[-1]

    spy50 = ma(spy,50)
    spy200 = ma(spy,200)

    qqq50 = ma(qqq,50)
    qqq200 = ma(qqq,200)

    breadth_data = breadth()

    sector_leaders = sector_leader_status()

    sectors = sector_strength()

    regime = classify(
        spy_close,
        spy50,
        spy200,
        qqq_close,
        qqq50,
        qqq200,
        vix_close,
        breadth_data["pct50"],
    )

    top3 = sectors[:3]
    bottom3 = sectors[-3:]

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines=[]

    lines.append(f"미국 시장 레짐 리포트 ({today})")
    lines.append("")
    lines.append(f"시장 판정: {regime}")
    lines.append("")

    lines.append("[지수 추세]")

    lines.append(f"SPY 현재가 {num(spy_close)}")
    lines.append(f"50MA {num(spy50)} / 상태 {'충족' if spy_close>spy50 else '미충족'}")
    lines.append(f"200MA {num(spy200)} / 상태 {'충족' if spy_close>spy200 else '미충족'}")

    lines.append("")

    lines.append(f"QQQ 현재가 {num(qqq_close)}")
    lines.append(f"50MA {num(qqq50)} / 상태 {'충족' if qqq_close>qqq50 else '미충족'}")
    lines.append(f"200MA {num(qqq200)} / 상태 {'충족' if qqq_close>qqq200 else '미충족'}")

    lines.append("")
    lines.append("[시장 Breadth]")

    lines.append(f"S&P500 50MA 위 비율 {pct(breadth_data['pct50'])} ({breadth_data['above50']}/{breadth_data['valid']})")
    lines.append(f"S&P500 200MA 위 비율 {pct(breadth_data['pct200'])} ({breadth_data['above200']}/{breadth_data['valid']})")

    lines.append("")
    lines.append("[신고가 / 신저가]")

    lines.append(f"52주 신고가 {breadth_data['high']}")
    lines.append(f"52주 신저가 {breadth_data['low']}")

    lines.append("")
    lines.append("[섹터 강도 상위 3개]")

    for r in top3:
        lines.append(f"{r['sector']} ({r['ticker']}) | 3M {pct(r['r3'])} | 6M {pct(r['r6'])}")

    lines.append("")
    lines.append("[섹터 강도 하위 3개]")

    for r in bottom3:
        lines.append(f"{r['sector']} ({r['ticker']}) | 3M {pct(r['r3'])} | 6M {pct(r['r6'])}")

    lines.append("")
    lines.append("[VIX]")

    lines.append(f"현재값 {num(vix_close)}")

    lines.append("")
    lines.append("[섹터 대표 종목 상태]")

    for r in sector_leaders:

        lines.append(
            f"{r['name']} ({r['ticker']}, {r['sector']}) "
            f"| 50MA {'위' if r['a50'] else '아래'} "
            f"| 200MA {'위' if r['a200'] else '아래'} "
            f"| 3M {pct(r['ret3'])}"
        )

    return "\n".join(lines)


# =========================================
# 실행
# =========================================

def main():

    report = build()

    send_telegram(report)

    print(report)


if __name__=="__main__":
    main()
