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
BREADTH_LOOKBACK = 252
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

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def send_telegram_message(text: str) -> None:
