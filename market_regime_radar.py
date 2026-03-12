import os
from datetime import datetime, timezone

import requests
import yfinance as yf
import pandas as pd


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# 기준
PRICE_HISTORY_PERIOD = "1y"
VIX_PERIOD = "6mo"

# 시장 레짐 기준
SPY_6M_RETURN_MIN = 0.0
QQQ_6M_RETURN_MIN = 0.0
VIX_STOP_LEVEL = 30.0
VIX_CAUTION_LEVEL = 22.0

# 아시아/서울 기준 날짜 표기
KST = timezone.utc


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram 환경변수 없음")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    response = requests.post(url, data=payload, timeout=20)
    print(f"[TELEGRAM] status={response.status_code} body={response.text[:300]}")


def download_close_series(ticker: str, period: str) -> pd.Series:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"{ticker} 다운로드 실패")

    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", ticker)] if ("Close", ticker) in df.columns else df["Close"]
    else:
        close = df["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise RuntimeError(f"{ticker} 종가 데이터 비어 있음")
    return close


def latest_trading_dates(series: pd.Series, n: int = 3) -> list[str]:
    idx = series.dropna().index[-n:]
    return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in idx]


def moving_average(series: pd.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    value = series.rolling(window).mean().iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def calc_return(series: pd.Series, lookback: int) -> float | None:
    if len(series) <= lookback:
        return None
    start = float(series.iloc[-lookback - 1])
    end = float(series.iloc[-1])
    if start <= 0:
        return None
    return (end / start) - 1.0


def pct_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def num_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def bool_text(value: bool) -> str:
    return "충족" if value else "미충족"


def classify_market(
    spy_close: float,
    spy_ma50: float | None,
    spy_ma200: float | None,
    qqq_close: float,
    qqq_ma50: float | None,
    qqq_ma200: float | None,
    spy_ret_6m: float | None,
    qqq_ret_6m: float | None,
    vix_close: float,
) -> tuple[str, str, list[str]]:
    reasons: list[str] = []

    spy_above_200 = spy_ma200 is not None and spy_close > spy_ma200
    spy_50_above_200 = spy_ma50 is not None and spy_ma200 is not None and spy_ma50 > spy_ma200
    qqq_above_200 = qqq_ma200 is not None and qqq_close > qqq_ma200
    qqq_50_above_200 = qqq_ma50 is not None and qqq_ma200 is not None and qqq_ma50 > qqq_ma200
    spy_6m_ok = spy_ret_6m is not None and spy_ret_6m > SPY_6M_RETURN_MIN
    qqq_6m_ok = qqq_ret_6m is not None and qqq_ret_6m > QQQ_6M_RETURN_MIN

    if not spy_above_200:
        reasons.append("SPY가 200일선 아래")
    if not qqq_above_200:
        reasons.append("QQQ가 200일선 아래")
    if not spy_50_above_200:
        reasons.append("SPY 50일선이 200일선 위가 아님")
    if not qqq_50_above_200:
        reasons.append("QQQ 50일선이 200일선 위가 아님")
    if not spy_6m_ok:
        reasons.append("SPY 최근 6개월 수익률이 0 이하")
    if not qqq_6m_ok:
        reasons.append("QQQ 최근 6개월 수익률이 0 이하")
    if vix_close >= VIX_STOP_LEVEL:
        reasons.append(f"VIX가 {VIX_STOP_LEVEL:.0f} 이상")

    stop = (
        (not spy_above_200)
        or (not qqq_above_200)
        or (not spy_50_above_200)
        or (not qqq_50_above_200)
        or (vix_close >= VIX_STOP_LEVEL)
    )

    if stop:
        return (
            "STOP",
            "오늘은 신규 돌파 매매 비추천",
            reasons,
        )

    caution = (
        (spy_ma50 is not None and spy_close <= spy_ma50)
        or (qqq_ma50 is not None and qqq_close <= qqq_ma50)
        or (not spy_6m_ok)
        or (not qqq_6m_ok)
        or (vix_close >= VIX_CAUTION_LEVEL)
    )

    if caution:
        if spy_ma50 is not None and spy_close <= spy_ma50:
            reasons.append("SPY가 50일선 아래 또는 근접")
        if qqq_ma50 is not None and qqq_close <= qqq_ma50:
            reasons.append("QQQ가 50일선 아래 또는 근접")
        if vix_close >= VIX_CAUTION_LEVEL:
            reasons.append(f"VIX가 {VIX_CAUTION_LEVEL:.0f} 이상으로 변동성 높음")

        return (
            "CAUTION",
            "오늘은 선별적 매매만 권장",
            reasons,
        )

    return (
        "GO",
        "오늘은 돌파 매매 가능한 시장",
        ["장기 추세와 중기 추세가 모두 양호"],
    )


def build_message() -> str:
    spy = download_close_series("SPY", PRICE_HISTORY_PERIOD)
    qqq = download_close_series("QQQ", PRICE_HISTORY_PERIOD)
    vix = download_close_series("^VIX", VIX_PERIOD)

    spy_close = float(spy.iloc[-1])
    qqq_close = float(qqq.iloc[-1])
    vix_close = float(vix.iloc[-1])

    spy_ma50 = moving_average(spy, 50)
    spy_ma200 = moving_average(spy, 200)
    qqq_ma50 = moving_average(qqq, 50)
    qqq_ma200 = moving_average(qqq, 200)

    spy_ret_6m = calc_return(spy, 126)
    qqq_ret_6m = calc_return(qqq, 126)

    regime, action, reasons = classify_market(
        spy_close=spy_close,
        spy_ma50=spy_ma50,
        spy_ma200=spy_ma200,
        qqq_close=qqq_close,
        qqq_ma50=qqq_ma50,
        qqq_ma200=qqq_ma200,
        spy_ret_6m=spy_ret_6m,
        qqq_ret_6m=qqq_ret_6m,
        vix_close=vix_close,
    )

    trading_dates = latest_trading_dates(spy, 3)

    spy_above_50 = spy_ma50 is not None and spy_close > spy_ma50
    spy_above_200 = spy_ma200 is not None and spy_close > spy_ma200
    qqq_above_50 = qqq_ma50 is not None and qqq_close > qqq_ma50
    qqq_above_200 = qqq_ma200 is not None and qqq_close > qqq_ma200

    kst_now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "미국 시장 레짐 리포트",
        f"리포트 생성 시각(UTC): {kst_now}",
        "",
        "최근 3개 거래일",
        f"- {trading_dates[0]}",
        f"- {trading_dates[1]}",
        f"- {trading_dates[2]}",
        "",
        "SPY",
        f"- 현재가: {num_text(spy_close)}",
        f"- 50일선: {num_text(spy_ma50)} / 상태: {bool_text(spy_above_50)}",
        f"- 200일선: {num_text(spy_ma200)} / 상태: {bool_text(spy_above_200)}",
        f"- 최근 6개월 수익률: {pct_text(spy_ret_6m)}",
        "",
        "QQQ",
        f"- 현재가: {num_text(qqq_close)}",
        f"- 50일선: {num_text(qqq_ma50)} / 상태: {bool_text(qqq_above_50)}",
        f"- 200일선: {num_text(qqq_ma200)} / 상태: {bool_text(qqq_above_200)}",
        f"- 최근 6개월 수익률: {pct_text(qqq_ret_6m)}",
        "",
        "VIX",
        f"- 현재값: {num_text(vix_close)}",
        f"- 해석: 22 이상이면 주의, 30 이상이면 신규 매매 보수적 접근",
        "",
        "최종 판정",
        f"- 시장 레짐: {regime}",
        f"- 실행 해석: {action}",
        "",
        "한글 해석",
        "- 50일선 위 = 중기 추세 양호",
        "- 200일선 위 = 장기 추세 양호",
        "- 최근 6개월 수익률 > 0 = 상승 모멘텀 유지",
        "- VIX 상승 = 변동성 확대, 돌파 실패 위험 증가",
        "",
        "판정 근거",
    ]

    for reason in reasons:
        lines.append(f"- {reason}")

    return "\n".join(lines)


def main() -> None:
    message = build_message()
    send_telegram_message(message)
    print(message)


if __name__ == "__main__":
    main()
