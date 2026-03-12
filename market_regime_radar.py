import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests
import yfinance as yf


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PRICE_HISTORY_PERIOD = "1y"
VIX_PERIOD = "6mo"
RET_3M_LOOKBACK = 63
RET_6M_LOOKBACK = 126

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

LEADER_TICKERS = {
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Meta": "META",
    "Apple": "AAPL",
}

SP500_UNIVERSE_CSV = "data/sp500_universe.csv"


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

    response = requests.post(url, json=payload, timeout=30)
    print(f"[TELEGRAM] status={response.status_code} body={response.text[:300]}")


def safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def pct_text(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def num_text(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def bool_text(value: bool) -> str:
    return "충족" if value else "미충족"


def latest_trading_dates(series: pd.Series, n: int = 3) -> list[str]:
    idx = series.dropna().index[-n:]
    return [pd.Timestamp(x).strftime("%Y-%m-%d") for x in idx]


def moving_average(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return safe_float(series.rolling(window).mean().iloc[-1])


def calc_return(series: pd.Series, lookback: int) -> Optional[float]:
    if len(series) <= lookback:
        return None
    start = safe_float(series.iloc[-lookback - 1])
    end = safe_float(series.iloc[-1])
    if start is None or end is None or start <= 0:
        return None
    return (end / start) - 1.0


def flatten_field(downloaded: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    if downloaded is None or downloaded.empty:
        return pd.Series(dtype="float64")

    if isinstance(downloaded.columns, pd.MultiIndex):
        if (field, ticker) in downloaded.columns:
            out = downloaded[(field, ticker)]
        elif (ticker, field) in downloaded.columns:
            out = downloaded[(ticker, field)]
        else:
            try:
                field_block = downloaded.xs(field, axis=1, level=0)
                if ticker in field_block.columns:
                    out = field_block[ticker]
                else:
                    return pd.Series(dtype="float64")
            except Exception:
                return pd.Series(dtype="float64")
    else:
        if field not in downloaded.columns:
            return pd.Series(dtype="float64")
        out = downloaded[field]

    out = pd.to_numeric(out, errors="coerce").dropna()
    out.name = ticker
    return out


def flatten_close(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    return flatten_field(downloaded, ticker, "Close")


def flatten_volume(downloaded: pd.DataFrame, ticker: str) -> pd.Series:
    return flatten_field(downloaded, ticker, "Volume")


def download_close_series(ticker: str, period: str) -> pd.Series:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    close = flatten_close(df, ticker)
    if close.empty:
        raise RuntimeError(f"{ticker} 다운로드 실패")
    return close


def get_sp500_tickers() -> list[str]:
    if not os.path.exists(SP500_UNIVERSE_CSV):
        raise FileNotFoundError(f"S&P500 유니버스 파일 없음: {SP500_UNIVERSE_CSV}")

    df = pd.read_csv(SP500_UNIVERSE_CSV)
    if "ticker" not in df.columns:
        raise ValueError("sp500_universe.csv에는 ticker 컬럼이 필요")

    tickers = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return sorted(set(tickers))


def calculate_breadth_high_low_adline_volume() -> dict:
    tickers = get_sp500_tickers()

    downloaded = yf.download(
        tickers=tickers,
        period=PRICE_HISTORY_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    close_map: dict[str, pd.Series] = {}
    volume_map: dict[str, pd.Series] = {}

    for ticker in tickers:
        close_series = flatten_close(downloaded, ticker)
        volume_series = flatten_volume(downloaded, ticker)

        if len(close_series) >= 200:
            close_map[ticker] = close_series
            if not volume_series.empty:
                volume_map[ticker] = volume_series

    if not close_map:
        return {
            "count_total": len(tickers),
            "count_valid": 0,
            "above_50": 0,
            "above_200": 0,
            "above_50_pct": None,
            "above_200_pct": None,
            "new_high_252": 0,
            "new_low_252": 0,
            "ad_line_latest": 0,
            "ad_line_20d_change": None,
            "up_volume": None,
            "down_volume": None,
            "up_down_volume_ratio": None,
        }

    above_50 = 0
    above_200 = 0
    new_high_252 = 0
    new_low_252 = 0

    closes_df = pd.DataFrame(close_map).sort_index()
    volumes_df = pd.DataFrame(volume_map).sort_index()

    for series in close_map.values():
        last = safe_float(series.iloc[-1])
        ma50 = moving_average(series, 50)
        ma200 = moving_average(series, 200)

        if last is not None and ma50 is not None and last > ma50:
            above_50 += 1
        if last is not None and ma200 is not None and last > ma200:
            above_200 += 1

        if len(series) >= 252:
            recent = series.iloc[-252:]
            high_252 = safe_float(recent.max())
            low_252 = safe_float(recent.min())
            if last is not None and high_252 is not None and last >= high_252 * 0.999:
                new_high_252 += 1
            if last is not None and low_252 is not None and last <= low_252 * 1.001:
                new_low_252 += 1

    count_valid = len(close_map)

    ad_line_latest = 0
    ad_line_20d_change = None
    if not closes_df.empty and len(closes_df) >= 25:
        diff = closes_df.diff()
        adv_decl = (diff > 0).astype(int) - (diff < 0).astype(int)
        daily_ad = adv_decl.sum(axis=1)
        ad_line = daily_ad.cumsum()
        ad_line_latest = int(ad_line.iloc[-1])
        ad_line_20d_change = safe_float(ad_line.iloc[-1] - ad_line.iloc[-21]) if len(ad_line) >= 21 else None

    up_volume = None
    down_volume = None
    up_down_volume_ratio = None
    if not closes_df.empty and not volumes_df.empty and len(closes_df) >= 2:
        last_close = closes_df.iloc[-1]
        prev_close = closes_df.iloc[-2]
        last_volume = volumes_df.iloc[-1].reindex(closes_df.columns)

        up_mask = last_close > prev_close
        down_mask = last_close < prev_close

        up_volume = safe_float(last_volume[up_mask].sum())
        down_volume = safe_float(last_volume[down_mask].sum())

        if up_volume is not None and down_volume is not None and down_volume > 0:
            up_down_volume_ratio = up_volume / down_volume

    return {
        "count_total": len(tickers),
        "count_valid": count_valid,
        "above_50": above_50,
        "above_200": above_200,
        "above_50_pct": above_50 / count_valid if count_valid else None,
        "above_200_pct": above_200 / count_valid if count_valid else None,
        "new_high_252": new_high_252,
        "new_low_252": new_low_252,
        "ad_line_latest": ad_line_latest,
        "ad_line_20d_change": ad_line_20d_change,
        "up_volume": up_volume,
        "down_volume": down_volume,
        "up_down_volume_ratio": up_down_volume_ratio,
    }


def get_sector_strength() -> list[dict]:
    rows = []

    for sector_name, ticker in SECTOR_ETFS.items():
        series = download_close_series(ticker, PRICE_HISTORY_PERIOD)
        ret_3m = calc_return(series, RET_3M_LOOKBACK)
        ret_6m = calc_return(series, RET_6M_LOOKBACK)

        score = 0.0
        if ret_3m is not None:
            score += ret_3m * 0.45
        if ret_6m is not None:
            score += ret_6m * 0.55

        rows.append(
            {
                "sector": sector_name,
                "ticker": ticker,
                "ret_3m": ret_3m,
                "ret_6m": ret_6m,
                "score": score,
            }
        )

    return sorted(rows, key=lambda x: x["score"], reverse=True)


def get_leader_health() -> list[dict]:
    rows = []

    for name, ticker in LEADER_TICKERS.items():
        series = download_close_series(ticker, PRICE_HISTORY_PERIOD)
        close = safe_float(series.iloc[-1])
        ma50 = moving_average(series, 50)
        ma200 = moving_average(series, 200)
        ret_3m = calc_return(series, RET_3M_LOOKBACK)

        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "close": close,
                "ma50": ma50,
                "ma200": ma200,
                "above_50": close is not None and ma50 is not None and close > ma50,
                "above_200": close is not None and ma200 is not None and close > ma200,
                "ret_3m": ret_3m,
            }
        )

    return rows


def interpret_breadth_50(pct: Optional[float]) -> str:
    if pct is None:
        return "데이터 부족"
    if pct >= 0.60:
        return "시장 내부 강도 양호"
    if pct >= 0.45:
        return "보통 수준, 약화 전환 여부 관찰"
    if pct >= 0.30:
        return "시장 내부 약화, 선별 필요"
    return "시장 내부 붕괴 가능성 높음"


def interpret_breadth_200(pct: Optional[float]) -> str:
    if pct is None:
        return "데이터 부족"
    if pct >= 0.70:
        return "장기 추세가 살아 있는 종목이 많음"
    if pct >= 0.50:
        return "장기 추세는 유지되지만 폭은 좁아짐"
    return "장기 추세 종목이 많이 무너진 상태"


def interpret_high_low(new_highs: int, new_lows: int) -> str:
    if new_highs > new_lows * 2:
        return "리더 확장 국면"
    if new_highs >= new_lows:
        return "신고가 우위, 아직 양호"
    if new_lows > new_highs * 2 and new_lows >= 40:
        return "내부 붕괴 가능성 높음"
    return "신고가보다 신저가가 많아 내부 약화 신호"


def interpret_ad_line(change_20d: Optional[float]) -> str:
    if change_20d is None:
        return "데이터 부족"
    if change_20d > 0:
        return "최근 20거래일 기준 상승 종목 누적 우위"
    if change_20d < 0:
        return "최근 20거래일 기준 하락 종목 누적 우위"
    return "최근 20거래일 기준 중립"


def interpret_up_down_volume(ratio: Optional[float]) -> str:
    if ratio is None:
        return "데이터 부족"
    if ratio >= 1.3:
        return "상승 거래량 우위, 자금 유입 양호"
    if ratio >= 0.8:
        return "거래량 흐름 중립"
    return "하락 거래량 우위, 분배 가능성 주의"


def interpret_leader_health(rows: list[dict]) -> str:
    if not rows:
        return "데이터 부족"

    above_50_count = sum(1 for r in rows if r["above_50"])
    above_200_count = sum(1 for r in rows if r["above_200"])

    if above_50_count >= 4 and above_200_count >= 5:
        return "대표 리더 구조 양호"
    if above_200_count >= 4 and above_50_count >= 2:
        return "리더는 장기 구조 유지, 단기 흔들림 존재"
    return "대표 리더 약화, 추세 전환 가능성 주의"


def classify_market(
    spy_close: float,
    spy_ma50: Optional[float],
    spy_ma200: Optional[float],
    qqq_close: float,
    qqq_ma50: Optional[float],
    qqq_ma200: Optional[float],
    spy_ret_6m: Optional[float],
    qqq_ret_6m: Optional[float],
    vix_close: float,
    breadth_50_pct: Optional[float],
    breadth_200_pct: Optional[float],
    new_highs: int,
    new_lows: int,
    ad_line_20d_change: Optional[float],
    up_down_volume_ratio: Optional[float],
    leader_rows: list[dict],
) -> tuple[str, str, list[str]]:
    reasons: list[str] = []

    spy_above_50 = spy_ma50 is not None and spy_close > spy_ma50
    spy_above_200 = spy_ma200 is not None and spy_close > spy_ma200
    spy_50_above_200 = spy_ma50 is not None and spy_ma200 is not None and spy_ma50 > spy_ma200

    qqq_above_50 = qqq_ma50 is not None and qqq_close > qqq_ma50
    qqq_above_200 = qqq_ma200 is not None and qqq_close > qqq_ma200
    qqq_50_above_200 = qqq_ma50 is not None and qqq_ma200 is not None and qqq_ma50 > qqq_ma200

    leader_above_50_count = sum(1 for r in leader_rows if r["above_50"])
    leader_above_200_count = sum(1 for r in leader_rows if r["above_200"])

    stop = False
    caution = False

    if not spy_above_200:
        reasons.append("SPY가 200일선 아래")
        stop = True
    if not qqq_above_200:
        reasons.append("QQQ가 200일선 아래")
        stop = True
    if not spy_50_above_200:
        reasons.append("SPY 50일선이 200일선 위가 아님")
        caution = True
    if not qqq_50_above_200:
        reasons.append("QQQ 50일선이 200일선 위가 아님")
        caution = True

    if spy_ret_6m is not None and spy_ret_6m <= 0:
        reasons.append("SPY 최근 6개월 수익률이 0 이하")
        caution = True
    if qqq_ret_6m is not None and qqq_ret_6m <= 0:
        reasons.append("QQQ 최근 6개월 수익률이 0 이하")
        caution = True

    if breadth_50_pct is not None and breadth_50_pct < 0.30:
        reasons.append("S&P500 50일선 위 종목 비율이 30% 미만")
        stop = True
    elif breadth_50_pct is not None and breadth_50_pct < 0.45:
        reasons.append("S&P500 50일선 위 종목 비율이 45% 미만")
        caution = True

    if breadth_200_pct is not None and breadth_200_pct < 0.50:
        reasons.append("S&P500 200일선 위 종목 비율이 50% 미만")
        caution = True

    if new_lows > new_highs * 2 and new_lows >= 40:
        reasons.append("신저가 종목 수가 신고가 종목 수보다 크게 많음")
        stop = True
    elif new_lows > new_highs:
        reasons.append("신저가 종목 수가 신고가 종목 수보다 많음")
        caution = True

    if ad_line_20d_change is not None and ad_line_20d_change < 0:
        reasons.append("A/D Line 20거래일 변화가 음수")
        caution = True

    if up_down_volume_ratio is not None and up_down_volume_ratio < 0.8:
        reasons.append("하락 거래량 우위")
        caution = True

    if leader_above_200_count <= 2:
        reasons.append("대표 리더 다수가 200일선 아래")
        stop = True
    elif leader_above_50_count <= 2:
        reasons.append("대표 리더 다수가 50일선 아래")
        caution = True

    if vix_close >= 30:
        reasons.append("VIX가 30 이상으로 변동성 매우 높음")
        stop = True
    elif vix_close >= 22:
        reasons.append("VIX가 22 이상으로 변동성 높음")
        caution = True

    if stop:
        return "STOP", "오늘은 신규 돌파 매매 비추천", reasons

    if caution or not spy_above_50 or not qqq_above_50:
        if not spy_above_50:
            reasons.append("SPY가 50일선 아래 또는 근접")
        if not qqq_above_50:
            reasons.append("QQQ가 50일선 아래 또는 근접")
        return "CAUTION", "오늘은 선별적 매매만 권장", reasons

    return "GO", "오늘은 돌파 매매 가능한 시장", ["장기 추세와 중기 추세, 내부 지표가 모두 양호"]


def build_message() -> str:
    spy = download_close_series("SPY", PRICE_HISTORY_PERIOD)
    qqq = download_close_series("QQQ", PRICE_HISTORY_PERIOD)
    vix = download_close_series("^VIX", VIX_PERIOD)

    spy_close = safe_float(spy.iloc[-1]) or 0.0
    qqq_close = safe_float(qqq.iloc[-1]) or 0.0
    vix_close = safe_float(vix.iloc[-1]) or 0.0

    spy_ma50 = moving_average(spy, 50)
    spy_ma200 = moving_average(spy, 200)
    qqq_ma50 = moving_average(qqq, 50)
    qqq_ma200 = moving_average(qqq, 200)

    spy_ret_6m = calc_return(spy, RET_6M_LOOKBACK)
    qqq_ret_6m = calc_return(qqq, RET_6M_LOOKBACK)

    breadth = calculate_breadth_high_low_adline_volume()
    sector_rows = get_sector_strength()
    leader_rows = get_leader_health()

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
        breadth_50_pct=breadth["above_50_pct"],
        breadth_200_pct=breadth["above_200_pct"],
        new_highs=breadth["new_high_252"],
        new_lows=breadth["new_low_252"],
        ad_line_20d_change=breadth["ad_line_20d_change"],
        up_down_volume_ratio=breadth["up_down_volume_ratio"],
        leader_rows=leader_rows,
    )

    trading_dates = latest_trading_dates(spy, 3)
    report_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    top3 = sector_rows[:3]
    bottom3 = sector_rows[-3:]

    lines = [
        "미국 시장 레짐 리포트",
        f"리포트 생성 시각(UTC): {report_time}",
        "",
        "최근 3개 거래일",
        f"- {trading_dates[0]}",
        f"- {trading_dates[1]}",
        f"- {trading_dates[2]}",
        "",
        "지수 추세",
        "SPY",
        f"- 현재가: {num_text(spy_close)}",
        f"- 50일선: {num_text(spy_ma50)} / 상태: {bool_text(spy_ma50 is not None and spy_close > spy_ma50)}",
        f"- 200일선: {num_text(spy_ma200)} / 상태: {bool_text(spy_ma200 is not None and spy_close > spy_ma200)}",
        f"- 최근 6개월 수익률: {pct_text(spy_ret_6m)}",
        "",
        "QQQ",
        f"- 현재가: {num_text(qqq_close)}",
        f"- 50일선: {num_text(qqq_ma50)} / 상태: {bool_text(qqq_ma50 is not None and qqq_close > qqq_ma50)}",
        f"- 200일선: {num_text(qqq_ma200)} / 상태: {bool_text(qqq_ma200 is not None and qqq_close > qqq_ma200)}",
        f"- 최근 6개월 수익률: {pct_text(qqq_ret_6m)}",
        "",
        "시장 Breadth",
        f"- S&P500 50일선 위 종목 비율: {pct_text(breadth['above_50_pct'])} ({breadth['above_50']}/{breadth['count_valid']})",
        f"- 해석: {interpret_breadth_50(breadth['above_50_pct'])}",
        f"- S&P500 200일선 위 종목 비율: {pct_text(breadth['above_200_pct'])} ({breadth['above_200']}/{breadth['count_valid']})",
        f"- 해석: {interpret_breadth_200(breadth['above_200_pct'])}",
        "",
        "신고가 / 신저가",
        f"- 52주 신고가 종목 수: {breadth['new_high_252']}",
        f"- 52주 신저가 종목 수: {breadth['new_low_252']}",
        f"- 해석: {interpret_high_low(breadth['new_high_252'], breadth['new_low_252'])}",
        "",
        "A/D Line",
        f"- 현재 누적값: {breadth['ad_line_latest']}",
        f"- 최근 20거래일 변화: {num_text(breadth['ad_line_20d_change'])}",
        f"- 해석: {interpret_ad_line(breadth['ad_line_20d_change'])}",
        "",
        "상승 거래량 / 하락 거래량",
        f"- 상승 거래량: {num_text(breadth['up_volume'])}",
        f"- 하락 거래량: {num_text(breadth['down_volume'])}",
        f"- 비율(상승/하락): {num_text(breadth['up_down_volume_ratio'])}",
        f"- 해석: {interpret_up_down_volume(breadth['up_down_volume_ratio'])}",
        "",
        "대표 리더 상태",
    ]

    for row in leader_rows:
        lines.append(
            f"- {row['name']} ({row['ticker']}) | 50일선 위 {bool_text(row['above_50'])} | "
            f"200일선 위 {bool_text(row['above_200'])} | 최근 3개월 {pct_text(row['ret_3m'])}"
        )

    lines.extend(
        [
            f"- 종합 해석: {interpret_leader_health(leader_rows)}",
            "",
            "섹터 강도 상위 3개",
        ]
    )

    for row in top3:
        lines.append(
            f"- {row['sector']} ({row['ticker']}) | 3개월 {pct_text(row['ret_3m'])} | 6개월 {pct_text(row['ret_6m'])}"
        )

    lines.extend(["", "섹터 강도 하위 3개"])

    for row in bottom3:
        lines.append(
            f"- {row['sector']} ({row['ticker']}) | 3개월 {pct_text(row['ret_3m'])} | 6개월 {pct_text(row['ret_6m'])}"
        )

    lines.extend(
        [
            "",
            "VIX",
            f"- 현재값: {num_text(vix_close)}",
            "- 해석: VIX가 높을수록 변동성이 크고 돌파 실패 위험이 커짐",
            "",
            "최종 판정",
            f"- 시장 레짐: {regime}",
            f"- 실행 해석: {action}",
            "",
            "판정 근거",
        ]
    )

    for reason in reasons:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "요약 해석",
            "- GO = 오늘 신규 돌파 매매 가능한 시장",
            "- CAUTION = 오늘은 선별적 매매만 권장",
            "- STOP = 오늘은 신규 돌파 매매 비추천",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    message = build_message()
    send_telegram_message(message)
    print(message)


if __name__ == "__main__":
    main()
