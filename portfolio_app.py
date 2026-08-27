import json
from pathlib import Path
import requests
import hashlib
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import gspread

# ---------------------------------------------------------------------------
# KONFIGURATION & GLOBALE VARIABLEN
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Portfolio-Analyse", page_icon="📊", layout="wide")

# Umrechnungsfaktor Gramm zu Unzen (1 Troy Unze = 31.1034768 Gramm)
OZ_TO_G = 31.1034768

# Custom-Styling-Funktion (beseitigt die schwere matplotlib-Abhängigkeit in der Cloud!)
def color_profit_loss(val):
    try:
        val = float(val)
        if pd.isna(val):
            return ''
        if val > 0:
            return 'background-color: rgba(46, 204, 113, 0.15); color: #2ecc71; font-weight: bold;'
        elif val < 0:
            return 'background-color: rgba(231, 76, 60, 0.15); color: #e74c3c; font-weight: bold;'
    except ValueError:
        pass
    return ''

# ---------------------------------------------------------------------------
# AUTO-SHUTDOWN BEI SCHLIESSEN DES BROWSERS (Nur lokal aktiv)
# ---------------------------------------------------------------------------
import os
import signal
import threading
from streamlit.runtime import get_instance

@st.cache_resource
def start_auto_shutdown_watchdog():
    def watchdog():
        time.sleep(10)  
        while True:
            time.sleep(2)
            try:
                runtime = get_instance()
                if runtime:
                    if len(runtime._session_mgr.list_active_sessions()) == 0:
                        os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()

# Watchdog nur starten, wenn wir lokal entwickeln (keine Secrets vorhanden)
if "gspread" not in st.secrets:
    start_auto_shutdown_watchdog()

APP_DIR = Path(__file__).resolve().parent
STORE_PATH = APP_DIR / "portfolio_store.csv"
TICKER_MAP_PATH = APP_DIR / "portfolio_ticker_map.json"
CASH_STORE_PATH = APP_DIR / "portfolio_cash.json"
BROKER_CSV_DIR = APP_DIR / "CSVs von Banken und Brokern"

BROKER_CSV_DIR.mkdir(parents=True, exist_ok=True)

# ISIN -> Yahoo-Finance-Ticker Standard-Mapping
DEFAULT_ISIN_TICKER_MAP = {
    "IE00B4L5Y983": "EUNL.DE",     # iShares Core MSCI World UCITS ETF (Acc)
    "IE00BKM4GZ66": "IS3N.DE",     # iShares Core MSCI EM IMI UCITS ETF (Acc)
    "IE0003XJA0J9": "WEBN.DE",     # Amundi Prime All Country World UCITS ETF (Acc)
    "DE0007664039": "VOW3.DE",     # Volkswagen (Vz.)
    "NL0000235190": "AIR.DE",      # Airbus
    "US5949181045": "MSFT",        # Microsoft
    "DK0062498333": "NOV.DE",      # Novo-Nordisk (B)
    "US8887871080": "TOST",        # Toast
    "US69608A1088": "PLTR",        # Palantir Technologies
    "DE0007030009": "RHM.DE",      # Rheinmetall
    "US4330001060": "HIMS",        # Hims & Hers Health
    "US3825501014": "GT",          # Goodyear Tire & Rubber
    "DE0005439004": "CON.DE",      # Continental
    "US98423F1093": "XMTR",        # Xometry
    "US64110L1061": "NFLX",        # Netflix
    "FR0010755611": "CL2.PA",      # MSCI USA 2x Lev
    "IE00BYWQWR46": "ESP0.DE",     # VanEck Video Gaming
    "US62914V1061": "NIO",         # NIO
    "CNE100000296": "BYDDF",       # BYD
    "US3364331070": "FSLR",        # First Solar
    "US88160R1014": "TSLA",        # Tesla
    "DE0006599905": "MRK.DE",      # Merck
    "LU1778762911": "SPOT",        # Spotify
    "US84615Q1031": "SPCX",        # SpaceX
    "BTC": "BTC-EUR",              # Bitcoin
}

# Statische Fallback-Holdings für bekannte ETFs
ISIN_HOLDINGS_FALLBACK = {
    "IE00B4L5Y983": [
        ("Apple Inc.", "AAPL", 0.049),
        ("Microsoft Corp.", "MSFT", 0.042),
        ("NVIDIA Corp.", "NVDA", 0.040),
        ("Amazon.com Inc.", "AMZN", 0.025),
        ("Meta Platforms Inc.", "META", 0.018),
        ("Broadcom Inc.", "AVGO", 0.016),
        ("Alphabet Inc. Class A", "GOOGL", 0.013),
        ("Alphabet Inc. Class C", "GOOG", 0.011),
        ("Tesla Inc.", "TSLA", 0.010),
        ("Berkshire Hathaway Inc. Class B", "BRK-B", 0.009),
    ],
    "IE00BKM4GZ66": [
        ("Taiwan Semiconductor Manufacturing", "TSM", 0.09),
        ("Tencent Holdings", "0700.HK", 0.04),
        ("Alibaba Group", "9988.HK", 0.025),
        ("Samsung Electronics", "005930.KS", 0.024),
        ("HDFC Bank", "HDB", 0.014),
        ("Reliance Industries", "RELIANCE.NS", 0.013),
        ("ICICI Bank", "IBN", 0.011),
        ("Meituan", "3690.HK", 0.008),
        ("Infosys", "INFY", 0.008),
        ("PDD Holdings", "PDD", 0.007),
    ],
    "IE0003XJA0J9": [
        ("Apple Inc.", "AAPL", 0.038),
        ("Microsoft Corp.", "MSFT", 0.034),
        ("NVIDIA Corp.", "NVDA", 0.032),
        ("Amazon.com Inc.", "AMZN", 0.020),
        ("Meta Platforms Inc.", "META", 0.014),
        ("Broadcom Inc.", "AVGO", 0.013),
        ("Alphabet Inc. Class A", "GOOGL", 0.011),
        ("Tesla Inc.", "TSLA", 0.008),
        ("Alphabet Inc. Class C", "GOOG", 0.008),
        ("Taiwan Semiconductor Manufacturing", "TSM", 0.007),
    ],
}

CACHE_TTL_PRICE = 60 * 15
CACHE_TTL_HISTORY = 60 * 60
CACHE_TTL_HOLDINGS = 60 * 60 * 24

SIMPLE_COLUMNS = {"isin", "name", "anteile", "kaufpreis", "datum"}
CANONICAL_COLUMNS = ["date", "ISIN", "Name", "type", "asset_class", "shares", "price", "amount", "fee", "tx_id"]

# ---------------------------------------------------------------------------
# UTILITY: WÄHRUNGSERKENNUNG (USD vs. EUR)
# ---------------------------------------------------------------------------

def is_usd_ticker(ticker: str) -> bool:
    t = str(ticker).strip()
    return "." not in t and not t.endswith("-EUR")

# ---------------------------------------------------------------------------
# YFINANCE: LIVE- & HISTORISCHE WECHSELKURSE (USD/EUR)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_usd_eur_rate() -> float:
    try:
        rate_data = yf.download("USDEUR=X", period="5d", interval="1d", progress=False)
        if not rate_data.empty:
            if isinstance(rate_data.columns, pd.MultiIndex):
                rate_data.columns = rate_data.columns.get_level_values(0)
            return float(rate_data["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 0.92

@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_historical_usd_eur_rate(start: str) -> pd.Series:
    try:
        rate_data = yf.download("USDEUR=X", start=start, interval="1d", progress=False)
        if not rate_data.empty:
            if isinstance(rate_data.columns, pd.MultiIndex):
                rate_data.columns = rate_data.columns.get_level_values(0)
            series = rate_data["Close"].dropna()
            series.index = pd.to_datetime(series.index).tz_localize(None)
            return series.ffill().bfill()
    except Exception:
        pass
    return pd.Series()

# ---------------------------------------------------------------------------
# MATH: MATHEMATISCHER XIRR / IZF SOLVER (Rein Python)
# ---------------------------------------------------------------------------

def xirr(cashflows: list[tuple[pd.Timestamp, float]]) -> float | None:
    if not cashflows:
        return None
    cashflows = [cf for cf in cashflows if cf[1] != 0]
    if len(cashflows) < 2:
        return None
    
    amounts = [cf[1] for cf in cashflows]
    if max(amounts) <= 0 or min(amounts) >= 0:
        return None
        
    t0 = min(cf[0] for cf in cashflows)
    
    def eq(r):
        val = 0.0
        for date, amt in cashflows:
            years = (date - t0).days / 365.25
            if 1.0 + r <= 0:
                val += amt * (1.0 + r) ** years if years >= 0 else 0
            else:
                val += amt / ((1.0 + r) ** years)
        return val

    low, high = -0.999, 10.0
    f_low = eq(low)
    f_high = eq(high)
    
    if np.sign(f_low) == np.sign(f_high):
        high = 50.0
        f_high = eq(high)
        if np.sign(f_low) == np.sign(f_high):
            return None

    for _ in range(150):
        mid = (low + high) / 2.0
        f_mid = eq(mid)
        if abs(f_mid) < 1e-4:
            return mid
        if np.sign(f_mid) == np.sign(f_low):
            low = mid
        else:
            high = mid
    return (low + high) / 2.0

# ---------------------------------------------------------------------------
# MATH: BERECHNUNG DER PERIODEN-PERFORMANCE AUS DER ZEITREIHE
# ---------------------------------------------------------------------------

def get_period_performance(value_history: pd.DataFrame, target_date: pd.Timestamp) -> tuple[float, float] | None:
    """Berechnet die absolute und prozentuale Depot-Performance für ein Zieldatum (Cashflow-bereinigt)."""
    if value_history.empty:
        return None
        
    today_date = value_history.index[-1]
    today_val = value_history["Portfolio-Wert"].iloc[-1]
    today_cap = value_history["Eingezahltes Kapital"].iloc[-1]
    
    past_dates = value_history.index[value_history.index <= target_date]
    if past_dates.empty:
        past_date = value_history.index[0]
    else:
        past_date = past_dates[-1]
        
    if past_date == today_date:
        return None
        
    past_val = value_history.loc[past_date, "Portfolio-Wert"]
    past_cap = value_history.loc[past_date, "Eingezahltes Kapital"]
    
    net_deposits = today_cap - past_cap
    pl_eur = today_val - net_deposits - past_val
    
    denominator = past_val + max(0.0, net_deposits)
    pl_pct = (pl_eur / denominator * 100) if denominator > 0 else 0.0
    
    return pl_eur, pl_pct

# ---------------------------------------------------------------------------
# UTILITY: NORMALIZE COMPANY NAMES (FÜR KONSOLIDIERTE DETAILANSICHT)
# ---------------------------------------------------------------------------

def normalize_company_name(name: str) -> str:
    """Bereinigt Firmennamen von Suffixen (AG, Inc, Corp, etc.), um sie zu gruppieren."""
    n = str(name).strip().lower()
    for suffix in [" ag", " inc", " corp", " co", " vz", " vzo", " vz.", " a/s", " class b", " (b)", " ag vzo"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()
    n = n.replace("&", "und").replace("  ", " ").strip()
    return n.title()

# ---------------------------------------------------------------------------
# DETEKTION & NORMALISIERUNG (UNTERSTÜTZT TR + SCALABLE)
# ---------------------------------------------------------------------------

def detect_format(df: pd.DataFrame) -> str:
    cols = {c.strip().lower() for c in df.columns}
    if SIMPLE_COLUMNS.issubset(cols):
        return "simple"
    if {"category", "type", "symbol", "shares", "amount"}.issubset(cols):
        return "transactions"  
    if {"status", "reference", "description", "assettype", "isin", "shares"}.issubset(cols):
        return "scalable"      
    raise ValueError("CSV-Format nicht erkannt.")

def normalize_transactions(df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if fmt == "simple":
        df["datum"] = pd.to_datetime(df["datum"])
        out = pd.DataFrame(
            {
                "date": df["datum"],
                "ISIN": df["isin"].astype(str),
                "Name": df["name"],
                "type": "BUY",
                "asset_class": "",
                "shares": pd.to_numeric(df["anteile"]),
                "price": pd.to_numeric(df["kaufpreis"]),
            }
        )
        out["amount"] = -(out["shares"] * out["price"])
        out["fee"] = 0.0
        out["tx_id"] = (
            "simple_"
            + out["ISIN"] + "_"
            + out["date"].dt.strftime("%Y%m%d") + "_"
            + out["shares"].round(6).astype(str) + "_"
            + out["price"].round(6).astype(str)
        )
        return out[CANONICAL_COLUMNS]

    if fmt == "scalable":
        df["date"] = pd.to_datetime(df["date"])
        
        mask = (df["status"] == "Executed") & (
            (df["assettype"] == "Security") | (df["type"] == "Distribution")
        )
        trades = df[mask & (df["isin"].notna() | (df["type"] == "Distribution"))].copy()
        
        if trades.empty:
            raise ValueError("Keine relevanten ausgeführten Transaktionen in der Scalable-CSV gefunden.")
            
        def parse_eu_num(val):
            if pd.isna(val):
                return 0.0
            val_str = str(val).strip()
            if not val_str:
                return 0.0
            if "," in val_str:
                val_str = val_str.replace(".", "").replace(",", ".")
            try:
                return float(val_str)
            except ValueError:
                return 0.0

        trades["shares"] = trades["shares"].apply(parse_eu_num)
        trades["price"] = trades["price"].apply(parse_eu_num)
        trades["amount"] = trades["amount"].apply(parse_eu_num)
        trades["fee"] = trades["fee"].apply(parse_eu_num)
        trades["tax"] = trades["tax"].apply(parse_eu_num)

        def map_scal_type(row):
            t = str(row["type"]).strip().lower()
            if t in ["buy", "savings plan"]:
                return "BUY"
            elif t == "sell":
                return "SELL"
            elif t == "security transfer":
                return "TRANSFER"
            elif t == "corporate action":
                return "CORP_ACTION"
            elif t == "distribution":
                return "DIVIDEND"  
            return "UNKNOWN"

        trades["canonical_type"] = trades.apply(map_scal_type, axis=1)
        trades = trades[trades["canonical_type"] != "UNKNOWN"].copy()

        def adjust_shares(row):
            sh = row["shares"]
            if row["canonical_type"] == "SELL":
                return -abs(sh)
            return sh
            
        trades["shares"] = trades.apply(adjust_shares, axis=1)

        out = pd.DataFrame(
            {
                "date": trades["date"],
                "ISIN": trades["isin"].astype(str),
                "Name": trades["description"],
                "type": trades["canonical_type"],
                "asset_class": "",
                "shares": trades["shares"],
                "price": trades["price"],
                "amount": trades["amount"],
                "fee": trades["fee"] + trades["tax"],  
                "tx_id": "scalable_" + trades["reference"].astype(str)
            }
        )
        return out[CANONICAL_COLUMNS]

    df["date"] = pd.to_datetime(df["date"])
    mask = (
        (df["category"] == "TRADING") & 
        (df["type"].isin(["BUY", "SELL", "REDEEM", "LIQUIDATION", "KNOCK_OUT"]))
    ) | (
        (df["category"] == "DELIVERY") & (df["type"] == "FREE_RECEIPT")
    )
    
    trades = df[mask & df["symbol"].notna() & (df["symbol"] != "")].copy()
    if trades.empty:
        raise ValueError("Keine relevanten TRADING/DELIVERY-Zeilen in der TR-CSV gefunden.")

    normalized_types = trades["type"].replace({
        "REDEEM": "SELL",
        "LIQUIDATION": "SELL",
        "KNOCK_OUT": "SELL"
    })

    out = pd.DataFrame(
        {
            "date": trades["date"],
            "ISIN": trades["symbol"].astype(str),
            "Name": trades["name"],
            "type": normalized_types,
            "asset_class": trades.get("asset_class", ""),
            "shares": pd.to_numeric(trades["shares"]),
            "price": pd.to_numeric(trades["price"], errors="coerce"),
            "amount": pd.to_numeric(trades["amount"], errors="coerce").fillna(0.0),
            "fee": pd.to_numeric(trades.get("fee", 0), errors="coerce").fillna(0.0),
            "tx_id": trades["transaction_id"].astype(str),
        }
    )
    return out[CANONICAL_COLUMNS]

# ---------------------------------------------------------------------------
# UTILITY: BROKER ZUORDNUNG FÜR CHART-DETAILS
# ---------------------------------------------------------------------------

def get_broker_name(tx_id: str) -> str:
    tx_str = str(tx_id)
    if tx_str.startswith("scalable_"):
        return "Scalable Capital"
    elif tx_str.startswith("simple_"):
        return "Manuelle Buchung"
    elif tx_str.startswith("virtual_"):
        return "Guthaben / Eigenbestand"
    else:
        return "Trade Republic"

# ---------------------------------------------------------------------------
# DEPOTÜBERTRAG-DIAGNOSE (INTELLIGENTER 2-STUFIGER ABGLEICH)
# ---------------------------------------------------------------------------

def identify_portfolio_transfers(tx: pd.DataFrame) -> tuple[set[str], dict[str, str]]:
    transfers_to_ignore = set()
    inbound_to_outbound = {}
    tx_sorted = tx.sort_values("date")
    
    inbounds = tx_sorted[(tx_sorted["type"] == "FREE_RECEIPT") | ((tx_sorted["type"] == "TRANSFER") & (tx_sorted["shares"] > 0))].copy()
    outbounds = tx_sorted[(tx_sorted["type"] == "TRANSFER") & (tx_sorted["shares"] < 0)].copy()
    
    matched_inbounds = set()
    matched_outbounds = set()
    
    for _, out_row in outbounds.iterrows():
        isin = out_row["ISIN"]
        qty_out = abs(out_row["shares"])
        date_out = out_row["date"]
        out_id = out_row["tx_id"]
        
        candidates = inbounds[
            (inbounds["ISIN"] == isin) & 
            (~inbounds["tx_id"].isin(matched_inbounds))
        ]
        
        for _, in_row in candidates.iterrows():
            qty_in = abs(in_row["shares"])
            date_in = in_row["date"]
            in_id = in_row["tx_id"]
            
            if abs(qty_in - qty_out) < 1e-4:
                days_diff = abs((date_in - date_out).days)
                if days_diff <= 45:
                    matched_inbounds.add(in_id)
                    matched_outbounds.add(out_id)
                    transfers_to_ignore.add(in_id)
                    transfers_to_ignore.add(out_id)
                    inbound_to_outbound[in_id] = out_id
                    break

    unmatched_inbounds = inbounds[~inbounds["tx_id"].isin(matched_inbounds)].sort_values("date")
    for _, row in unmatched_inbounds.iterrows():
        isin = row["ISIN"]
        qty_in = row["shares"]
        date_in = row["date"]
        tx_id = row["tx_id"]
        
        prior_buys = tx_sorted[(tx_sorted["ISIN"] == isin) & (tx_sorted["date"] < date_in) & (tx_sorted["type"] == "BUY")]["shares"].sum()
        prior_sells = abs(tx_sorted[(tx_sorted["ISIN"] == isin) & (tx_sorted["date"] < date_in) & (tx_sorted["type"] == "SELL")]["shares"].sum())
        prior_balance = prior_buys - prior_sells
        
        if prior_balance >= qty_in * 0.75 and qty_in > 1e-5:
            transfers_to_ignore.add(tx_id)
            inbound_to_outbound[tx_id] = "unmatched"
            
    return transfers_to_ignore, inbound_to_outbound

# ---------------------------------------------------------------------------
# MATH: KONSOLIDIERUNG VON AKTIENSPLITS (SAME-DAY SAME-ISIN CORP ACTIONS)
# ---------------------------------------------------------------------------

def pre_process_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    corp_actions = df[df["type"] == "CORP_ACTION"].copy()
    if corp_actions.empty:
        return df
        
    df_sorted = df.sort_values("date")
    transfers_to_ignore = set()
    all_new_rows = []
    
    for (date_val, isin), group in corp_actions.groupby(["date", "ISIN"]):
        if len(group) > 1:
            net_shares = group["shares"].sum()
            
            first_row = group.iloc[0].copy()
            first_row["type"] = "SPLIT"
            first_row["shares"] = net_shares
            first_row["amount"] = 0.0  
            first_row["price"] = 0.0
            first_row["fee"] = 0.0
            
            all_new_rows.append(first_row)
            transfers_to_ignore.update(group["tx_id"].tolist())
            
    df_clean = df_sorted[~df_sorted["tx_id"].isin(transfers_to_ignore)].copy()
    if all_new_rows:
        df_clean = pd.concat([df_clean, pd.DataFrame(all_new_rows)], ignore_index=True)
        
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    return df_clean.sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# MATH: ONLINE ISIN-TO-TICKER SEARCH (Yahoo Query API)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_HOLDINGS, show_spinner=False)
def resolve_isin_to_ticker_online(isin: str) -> str | None:
    url = 'https://query1.finance.yahoo.com/v1/finance/search'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36',
    }
    params = {
        'q': isin,
        'quotesCount': 1,
        'newsCount': 0,
        'listsCount': 0,
    }
    try:
        resp = requests.get(url=url, headers=headers, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if 'quotes' in data and len(data['quotes']) > 0:
                return data['quotes'][0]['symbol']
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# UTILITY: FILE HASH FÜR DUPLIKATSCHUTZ IM ORDNER
# ---------------------------------------------------------------------------

def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def is_duplicate_file(uploaded_bytes: bytes) -> bool:
    if not BROKER_CSV_DIR.exists():
        return False
    uploaded_hash = calculate_file_hash(uploaded_bytes)
    for file_path in BROKER_CSV_DIR.glob("*.csv"):
        try:
            existing_hash = calculate_file_hash(file_path.read_bytes())
            if uploaded_hash == existing_hash:
                return True
        except Exception:
            pass
    return False

# ---------------------------------------------------------------------------
# PERSISTENTE DATEN-HILFSFUNKTIONEN & SYSTEM-MIGRATION
# ---------------------------------------------------------------------------

def migrate_database():
    if STORE_PATH.exists():
        try:
            df = pd.read_csv(STORE_PATH)
            updated = False
            
            transfers = df[df["type"] == "TRANSFER"]
            if not transfers.empty:
                grouped = transfers.groupby(["ISIN", "date"])
                for (isin, date), group in grouped:
                    if len(group) > 1:
                        if (group["shares"] > 0).any() and (group["shares"] < 0).any():
                            df.loc[df["tx_id"].isin(group["tx_id"]), "type"] = "CORP_ACTION"
                            updated = True
                            
            if updated:
                df.to_csv(STORE_PATH, index=False)
        except Exception:
            pass

def load_ticker_overrides(all_isins: list) -> dict:
    mapping = {}
    if TICKER_MAP_PATH.exists():
        try:
            mapping = json.loads(TICKER_MAP_PATH.read_text(encoding="utf-8"))
        except Exception:
            mapping = {}
            
    updated = False
    for isin in all_isins:
        current_val = mapping.get(isin, "")
        if mapping.get(isin) == "WPEA.PA" and isin == "IE0003XJA0J9":
            mapping[isin] = "WEBN.DE"
            updated = True
        if mapping.get(isin) in ["NOVC.DE", ""] and isin == "DK0062498333":
            mapping[isin] = "NOV.DE"
            updated = True
        if mapping.get(isin) in ["", None]:
            if isin in DEFAULT_ISIN_TICKER_MAP:
                mapping[isin] = DEFAULT_ISIN_TICKER_MAP[isin]
                updated = True
            else:
                resolved_ticker = resolve_isin_to_ticker_online(isin)
                if resolved_ticker:
                    mapping[isin] = resolved_ticker
                    updated = True
                    
    if updated:
        save_ticker_overrides(mapping)
        
    return mapping

def save_ticker_overrides(mapping: dict) -> None:
    try:
        TICKER_MAP_PATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# YFINANCE ROBUSTER MULTI-INDEX PARSER & DOWNLOADERS
# ---------------------------------------------------------------------------

def extract_close_prices(data: pd.DataFrame, ticker: str) -> pd.Series:
    if data.empty:
        return pd.Series()
    
    cols = data.columns
    
    if not isinstance(cols, pd.MultiIndex):
        if ticker in cols:
            return data[ticker]
        if "Close" in cols:
            return data["Close"]
        return pd.Series()
        
    if "Close" in cols.levels[0] and ticker in cols.levels[1]:
        return data.xs(key=("Close", ticker), axis=1)
    if ("Close", ticker) in cols:
        return data[("Close", ticker)]
        
    if ticker in cols.levels[0] and "Close" in cols.levels[1]:
        return data.xs(key=(ticker, "Close"), axis=1)
    if (ticker, "Close") in cols:
        return data[(ticker, "Close")]
        
    flat_cols = list(cols)
    for col in flat_cols:
        if len(col) == 2:
            if col[0] == "Close" and col[1] == ticker:
                return data[col]
            if col[0] == ticker and col[1] == "Close":
                return data[col]
                
    if len(cols.levels[1]) == 1 and "Close" in cols.levels[0]:
        return data["Close"].squeeze()
        
    return pd.Series()

@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def fetch_current_prices(tickers: tuple) -> tuple[dict, str]:
    prices = {}
    update_time_str = pd.Timestamp.now(tz="Europe/Berlin").strftime("%d.%m.%Y %H:%M:%S")
    if not tickers:
        return prices, update_time_str
        
    usd_eur_rate = fetch_usd_eur_rate()
    
    try:
        data = yf.download(list(tickers), period="5d", interval="1d", progress=False)
    except Exception:
        data = pd.DataFrame()
        
    for ticker in tickers:
        try:
            series = pd.Series()
            if not data.empty:
                series = extract_close_prices(data, ticker).dropna()
                
            if series.empty:
                single_data = yf.download(ticker, period="5d", interval="1d", progress=False)
                if not single_data.empty:
                    series = extract_close_prices(single_data, ticker).dropna()
                    
            if not series.empty:
                price = float(series.iloc[-1])
                if is_usd_ticker(ticker):
                    price = price * usd_eur_rate
                prices[ticker] = price
            else:
                prices[ticker] = None
        except Exception:
            prices[ticker] = None
    return prices, update_time_str

@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_price_history(tickers: tuple, start: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    
    tickers = tuple(t for t in tickers if t)
    usd_eur_rates = fetch_historical_usd_eur_rate(start)
    
    try:
        data = yf.download(list(tickers), start=start, interval="1d", progress=False)
    except Exception:
        data = pd.DataFrame()
        
    frames = {}
    for ticker in tickers:
        try:
            series = pd.Series()
            if not data.empty:
                series = extract_close_prices(data, ticker)
                
            if series.empty or series.isna().all():
                single_data = yf.download(ticker, start=start, interval="1d", progress=False)
                if single_data.empty:
                    single_data = yf.download(ticker, interval="1d", progress=False)
                if not single_data.empty:
                    series = extract_close_prices(single_data, ticker)
                        
            if not series.empty:
                frames[ticker] = series
        except Exception:
            continue
                
    if not frames:
        return pd.DataFrame()
        
    hist = pd.DataFrame(frames)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist.ffill().bfill()
    
    if not usd_eur_rates.empty:
        rates_aligned = usd_eur_rates.reindex(hist.index).ffill().bfill()
        for ticker in hist.columns:
            if is_usd_ticker(ticker):
                hist[ticker] = hist[ticker] * rates_aligned
                
    return hist

@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def fetch_historical_price(ticker: str, date_str: str):
    try:
        target = pd.Timestamp(date_str)
        start = (target - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        end = (target + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        hist = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
        if hist.empty:
            return None
            
        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        series = extract_close_prices(hist, ticker)
        if series.empty:
            return None
            
        on_or_before = series.index[series.index <= target]
        chosen = on_or_before.max() if len(on_or_before) > 0 else series.index.min()
        close = series.loc[chosen]
        price = float(close)
        
        if is_usd_ticker(ticker):
            rate_hist = yf.download("USDEUR=X", start=start, end=end, interval="1d", progress=False)
            if not rate_hist.empty:
                rate_hist.index = pd.to_datetime(rate_hist.index).tz_localize(None)
                rate_series = extract_close_prices(rate_hist, "USDEUR=X")
                if not rate_series.empty:
                    on_or_before_rate = rate_series.index[rate_series.index <= target]
                    chosen_rate = on_or_before_rate.max() if len(on_or_before_rate) > 0 else rate_series.index.min()
                    rate = float(rate_series.loc[chosen_rate])
                    price = price * rate
                
        return price
    except Exception:
        return None

# ---------------------------------------------------------------------------
# ETF / FONDS INTERNE ANTEILE ABRUFEN
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_HOLDINGS, show_spinner=False)
def fetch_etf_holdings(ticker: str, isin: str):
    try:
        t = yf.Ticker(ticker)
        top = t.funds_data.top_holdings
        if top is not None and not top.empty:
            df = top.reset_index()
            df.columns = ["Symbol", "Name", "Weight"] if len(df.columns) == 3 else df.columns
            if "Holding Percent" in df.columns:
                df = df.rename(columns={"Holding Percent": "Weight"})
            df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
            df = df.dropna(subset=["Weight"])
            if not df.empty:
                return df[["Name", "Symbol", "Weight"]], "live (yfinance)"
    except Exception:
        pass

    if isin in ISIN_HOLDINGS_FALLBACK:
        df = pd.DataFrame(ISIN_HOLDINGS_FALLBACK[isin], columns=["Name", "Symbol", "Weight"])
        return df, "Fallback-Tabelle"

    return pd.DataFrame(columns=["Name", "Symbol", "Weight"]), "keine Daten"

# ---------------------------------------------------------------------------
# PORTFOLIO-LOGIK
# ---------------------------------------------------------------------------

def build_positions(tx: pd.DataFrame, ticker_map: dict):
    ignored_tx_ids, _ = identify_portfolio_transfers(tx)
    active_tx = tx[~tx["tx_id"].isin(ignored_tx_ids)].copy()

    active_tx = active_tx.sort_values("date")
    open_rows, closed_rows, unresolved = [], [], []

    for isin, group in active_tx.groupby("ISIN"):
        name = group["Name"].iloc[-1]
        asset_class = group["asset_class"].iloc[-1] if "asset_class" in group.columns else ""
        shares = 0.0
        invested = 0.0
        realized_pl = 0.0
        total_invested_ever = 0.0
        total_proceeds_ever = 0.0
        first_date = group["date"].min()
        last_date = group["date"].max()

        for _, row in group.iterrows():
            if row["type"] == "BUY":
                cost = abs(row["amount"]) + abs(row["fee"] or 0)
                shares += row["shares"]
                invested += cost
                total_invested_ever += cost

            elif row["type"] == "SELL":
                sold_qty = abs(row["shares"])
                proceeds = abs(row["amount"])
                fee_abs = abs(row["fee"] or 0)
                avg_cost = (invested / shares) if shares > 1e-9 else 0.0
                actual_sold = min(sold_qty, shares) if shares > 0 else sold_qty
                cost_removed = avg_cost * actual_sold
                invested -= cost_removed
                realized_pl += proceeds - fee_abs - cost_removed
                total_proceeds_ever += proceeds
                shares += row["shares"]

            elif row["type"] == "SPLIT":
                shares += row["shares"]

            elif row["type"] in ["FREE_RECEIPT", "TRANSFER", "CORP_ACTION"]:
                if row["shares"] > 0:
                    price = row["price"] if pd.notna(row["price"]) and row["price"] > 0 else None
                    if price is None:
                        ticker = ticker_map.get(isin)
                        price = fetch_historical_price(ticker, row["date"].strftime("%Y-%m-%d")) if ticker else None
                    
                    if price is not None:
                        cost = price * row["shares"]
                        invested += cost
                        total_invested_ever += cost
                    else:
                        unresolved.append((isin, name, row["date"], row["shares"]))
                    shares += row["shares"]
                else:
                    removed_qty = abs(row["shares"])
                    avg_cost = (invested / shares) if shares > 1e-9 else 0.0
                    actual_removed = min(removed_qty, shares) if shares > 0 else removed_qty
                    cost_removed = avg_cost * actual_removed
                    invested -= cost_removed
                    shares += row["shares"]

            if abs(shares) < 1e-6:
                shares = 0.0

        if shares > 1e-6 or shares < -1e-6:
            open_rows.append(
                {
                    "ISIN": isin,
                    "Name": name,
                    "asset_class": asset_class,
                    "shares": shares,
                    "invested": invested,
                    "avg_cost": (invested / shares) if abs(shares) > 1e-9 else 0.0,
                    "first_date": first_date,
                }
            )
        else:
            realized_pl_pct = (realized_pl / total_invested_ever * 100) if total_invested_ever else 0.0
            closed_rows.append(
                {
                    "ISIN": isin,
                    "Name": name,
                    "asset_class": asset_class,
                    "total_invested": total_invested_ever,
                    "total_proceeds": total_proceeds_ever,
                    "realized_pl": realized_pl,
                    "realized_pl_pct": realized_pl_pct,
                    "first_date": first_date,
                    "last_date": last_date,
                }
            )

    open_df = pd.DataFrame(open_rows) if open_rows else pd.DataFrame(columns=["ISIN", "Name", "asset_class", "shares", "invested", "avg_cost", "first_date"])
    closed_df = pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame(columns=["ISIN", "Name", "asset_class", "total_invested", "total_proceeds", "realized_pl", "realized_pl_pct", "first_date", "last_date"])

    return open_df, closed_df, unresolved


def build_portfolio_value_history(tx: pd.DataFrame, price_history: pd.DataFrame, ticker_map: dict) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame()

    tx = tx.copy()
    ignored_tx_ids, _ = identify_portfolio_transfers(tx)
    tx = tx[~tx["tx_id"].isin(ignored_tx_ids)].copy()

    # Berechne den exakten historischen Cash-Flow (inklusive Einstandswerte von Depotübertragungen)
    cash_flows = []
    for idx, row in tx.iterrows():
        if row["type"] in ["TRANSFER", "FREE_RECEIPT", "CORP_ACTION"]:
            if row["shares"] > 0:
                price = row["price"] if (pd.notna(row["price"]) and row["price"] > 0) else None
                if price is None:
                    ticker = ticker_map.get(row["ISIN"])
                    price = fetch_historical_price(ticker, row["date"].strftime("%Y-%m-%d")) if ticker else None
                
                # Einfließende Transfers erhöhen das eingezahlte Kapital um den realen Einstandswert!
                cost = (price * row["shares"]) if price is not None else 0.0
                cash_flows.append(cost)
            else:
                cash_flows.append(0.0)
        elif row["type"] == "SPLIT":
            cash_flows.append(0.0)
        else:
            cash_flows.append(-row["amount"])
            
    tx["cash_flow"] = cash_flows

    all_dates = price_history.index
    total_value = pd.Series(0.0, index=all_dates)
    invested_capital = pd.Series(0.0, index=all_dates)

    for isin, group in tx.groupby("ISIN"):
        if isin in ["Physisches Cash", "Andere Assets", "Offene Kredite"]:
            continue
            
        ticker = ticker_map.get(isin)
        if not ticker or ticker not in price_history.columns:
            continue

        daily = group.groupby("date")[["shares", "cash_flow"]].sum().sort_index()

        shares_over_time = daily["shares"].cumsum()
        shares_over_time = shares_over_time.reindex(all_dates.union(shares_over_time.index)).sort_index()
        shares_over_time = shares_over_time.ffill().fillna(0).reindex(all_dates).ffill().fillna(0)
        
        total_value = total_value.add((shares_over_time * price_history[ticker]).fillna(0), fill_value=0)

        cash_over_time = daily["cash_flow"].cumsum()
        cash_over_time = cash_over_time.reindex(all_dates.union(cash_over_time.index)).sort_index()
        cash_over_time = cash_over_time.ffill().fillna(0).reindex(all_dates).ffill().fillna(0)
        invested_capital = invested_capital.add(cash_over_time, fill_value=0)

    result = pd.DataFrame({"Portfolio-Wert": total_value, "Eingezahltes Kapital": invested_capital})
    return result[result.index >= tx["date"].min()]
# ---------------------------------------------------------------------------
# HYBRID DATABASE SCHNITTSTELLEN (MIT API-SCHONENDEM CACHING)
# ---------------------------------------------------------------------------

def is_using_gsheets() -> bool:
    return "gspread" in st.secrets

@st.cache_resource(show_spinner=False)
def get_gspread_client():
    """Authentifiziert sich einmalig bei Google (schont Ressourcen und beschleunigt Ladezeiten drastisch)."""
    try:
        creds = dict(st.secrets["gspread"])
        creds["private_key"] = creds["private_key"].replace("\\n", "\n")
        return gspread.service_account_from_dict(creds)
    except Exception as e:
        st.error(f"Fehler bei der Google Sheets Verbindung: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def get_spreadsheet_url(spreadsheet_name: str) -> str:
    """Ruft den exakten, dynamischen Link zum Google Sheet ab (gecached für blitzschnelle Ladezeiten)."""
    if is_using_gsheets():
        client = get_gspread_client()
        if client:
            try:
                sh = client.open(spreadsheet_name)
                return sh.url
            except Exception:
                pass
    return ""

@st.cache_data(show_spinner="Lade Transaktionen aus Google Sheets...", ttl=300)
def load_store_hybrid(spreadsheet_name: str) -> pd.DataFrame:
    if is_using_gsheets():
        client = get_gspread_client()
        if client:
            try:
                sh = client.open(spreadsheet_name)
                worksheet = sh.worksheet("portfolio_store")
                records = worksheet.get_all_records()
                if not records:
                    return pd.DataFrame(columns=CANONICAL_COLUMNS)
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                return df
            except Exception:
                return pd.DataFrame(columns=CANONICAL_COLUMNS)
    else:
        if STORE_PATH.exists():
            df = pd.read_csv(STORE_PATH)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

def save_store_hybrid(spreadsheet_name: str, df: pd.DataFrame) -> None:
    if is_using_gsheets():
        client = get_gspread_client()
        if client:
            try:
                sh = client.open(spreadsheet_name)
                worksheet = sh.worksheet("portfolio_store")
                worksheet.clear()
                
                # NaNs bereinigen (in leere Strings umwandeln), um JSON-Kompatibilität zu sichern
                df_write = df.copy()
                df_write = df_write.fillna("")
                df_write["date"] = df_write["date"].astype(str)
                
                worksheet.update([df_write.columns.values.tolist()] + df_write.values.tolist())
                load_store_hybrid.clear()
            except Exception as e:
                st.error(f"Fehler beim Speichern in Google Sheets: {e}")
    else:
        df.to_csv(STORE_PATH, index=False)
        load_store_hybrid.clear()

@st.cache_data(show_spinner="Lade Kontostände aus Google Sheets...", ttl=300)
def load_cash_values_hybrid(spreadsheet_name: str) -> dict:
    defaults = {
        "tagesgeld": 0.0, 
        "girokonto": 0.0, 
        "darlehen": 0.0, 
        "andere_assets": 0.0, 
        "cash_date": "2021-01-01", 
        "gold_unit": "Unzen (oz)",
        "gold_amount": 0.0, 
        "gold_cost": 0.0, 
        "gold_date": "2021-01-01",
        "silver_unit": "Unzen (oz)",
        "silver_amount": 0.0, 
        "silver_cost": 0.0, 
        "silver_date": "2021-01-01"
    }
    if is_using_gsheets():
        client = get_gspread_client()
        if client:
            try:
                sh = client.open(spreadsheet_name)
                worksheet = sh.worksheet("portfolio_cash")
                records = worksheet.get_all_records()
                if records:
                    data = records[0]
                    if "gold_ounces" in data and "gold_amount" not in data:
                        data["gold_amount"] = data["gold_ounces"]
                        data["gold_unit"] = "Unzen (oz)"
                    for k, v in defaults.items():
                        if k not in data:
                            data[k] = v
                    return data
            except Exception:
                pass
        return defaults
    else:
        if CASH_STORE_PATH.exists():
            try:
                data = json.loads(CASH_STORE_PATH.read_text(encoding="utf-8"))
                if "gold_ounces" in data and "gold_amount" not in data:
                    data["gold_amount"] = data["gold_ounces"]
                    data["gold_unit"] = "Unzen (oz)"
                for k, v in defaults.items():
                    if k not in data:
                        data[k] = v
                return data
            except Exception:
                return defaults
        return defaults

def save_cash_values_hybrid(spreadsheet_name: str, tagesgeld: float, girokonto: float, darlehen: float, andere_assets: float, cash_date: str, gold_unit: str, gold_amount: float, gold_cost: float, gold_date: str, silver_unit: str, silver_amount: float, silver_cost: float, silver_date: str) -> None:
    data = {
        "tagesgeld": tagesgeld,
        "girokonto": girokonto,
        "darlehen": darlehen,
        "andere_assets": andere_assets,
        "cash_date": cash_date,  # NEU: Speichert das Cash-Datum im Sheet
        "gold_unit": gold_unit,
        "gold_amount": gold_amount,
        "gold_cost": gold_cost,
        "gold_date": gold_date,
        "silver_unit": silver_unit,
        "silver_amount": silver_amount,
        "silver_cost": silver_cost,
        "silver_date": silver_date
    }
    if is_using_gsheets():
        client = get_gspread_client()
        if client:
            try:
                sh = client.open(spreadsheet_name)
                worksheet = sh.worksheet("portfolio_cash")
                worksheet.clear()
                df = pd.DataFrame([data])
                worksheet.update([df.columns.values.tolist()] + df.values.tolist())
                load_cash_values_hybrid.clear()
            except Exception as e:
                st.error(f"Fehler beim Speichern der Cash-Bestände in Google Sheets: {e}")
    else:
        try:
            CASH_STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
            load_cash_values_hybrid.clear()
        except Exception:
            pass

def merge_into_store_hybrid(spreadsheet_name: str, new_tx: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    store = load_store_hybrid(spreadsheet_name)
    if store.empty:
        merged = new_tx.copy()
        num_new = len(merged)
    else:
        combined = pd.concat([store, new_tx], ignore_index=True)
        merged = combined.drop_duplicates(subset="tx_id", keep="first")
        num_new = len(merged) - len(store)
    merged = merged.sort_values("date").reset_index(drop=True)
    save_store_hybrid(spreadsheet_name, merged)
    return merged, num_new

def process_broker_csv_folder_hybrid(spreadsheet_name: str) -> tuple[pd.DataFrame, int]:
    BROKER_CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_files = list(BROKER_CSV_DIR.glob("*.csv"))
    
    store_df = load_store_hybrid(spreadsheet_name)
    if not csv_files:
        return store_df, 0
        
    all_new_txs = []
    for file_path in csv_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                sample = f.read(2048)
            sep = ";" if ";" in sample and sample.count(";") > sample.count(",") else ","
            
            raw = pd.read_csv(file_path, sep=sep)
            fmt = detect_format(raw)
            new_tx = normalize_transactions(raw, fmt)
            all_new_txs.append(new_tx)
        except Exception:
            pass
            
    if all_new_txs:
        combined_new = pd.concat(all_new_txs, ignore_index=True)
        store_df, total_new = merge_into_store_hybrid(spreadsheet_name, combined_new)
        
        for file_path in csv_files:
            try:
                file_path.unlink()
            except Exception:
                pass
                
        return store_df, total_new
        
    return store_df, 0

# ---------------------------------------------------------------------------
# PORTFOLIO- & BENUTZERWAHL (STEUERUNG DER GOOGLE SHEET AUSWAHL)
# ---------------------------------------------------------------------------

spreadsheet_name = "Portfolio_Linus"

if is_using_gsheets():
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        owner_choice = st.selectbox(
            "Portfolio-Besitzer auswählen", 
            ["Linus", "Janic (Getrenntes Portfolio)"],
            help="Hier kannst du auswählen, welches Portfolio geladen werden soll. Beide liegen sicher getrennt auf Google Drive."
        )
    with col_sel2:
        pin = st.text_input("PIN zur Freischaltung", type="password", help="Bitte gib die PIN ein, um Zugriff auf das Portfolio zu erhalten.")
    
    if owner_choice == "Linus":
        if pin != "1809":
            st.warning("Bitte gib die korrekte PIN für Linus ein, um die Daten freizuschalten.")
            st.stop()
        spreadsheet_name = "Portfolio_Linus"
    else:
        if pin != "3112":
            st.warning("Bitte gib die korrekte PIN für Janic ein, um die Daten freizuschalten.")
            st.stop()
        spreadsheet_name = "Portfolio_Janic"
else:
    migrate_database()

# ---------------------------------------------------------------------------
# SIDEBAR 1: PORTFOLIO-DATEN & DATEI-UPLOAD (AM ANFANG DER APP INITIALISIERT)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Portfolio-Daten")
    
    # Dynamischen Direktlink zum Google Sheet des aktiven Benutzers abrufen & anzeigen
    spreadsheet_url = get_spreadsheet_url(spreadsheet_name)
    if spreadsheet_url:
        st.markdown(f"🟢 **[📂 Google Sheet öffnen]({spreadsheet_url})**")
        st.caption("Öffnet deine Excel-Datenbank direkt in Google Drive")
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Zeige Anzahl der gefundenen Dateien im Ordner
    scanned_files_count = len(list(BROKER_CSV_DIR.glob("*.csv")))
    st.caption(f"📁 Auto-Scan-Ordner: `{BROKER_CSV_DIR.name}`")
    st.caption(f"Gefundene CSV-Dateien: **{scanned_files_count}**")
    
    uploaded_file = st.file_uploader("CSV manuell hinzufügen (optional)", type=["csv"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        
        if is_duplicate_file(file_bytes):
            st.sidebar.warning("Diese CSV-Datei existiert bereits im Ordner.")
        else:
            target_path = BROKER_CSV_DIR / uploaded_file.name
            if target_path.exists():
                target_path = BROKER_CSV_DIR / f"{uploaded_file.name.replace('.csv', '')}_{int(time.time())}.csv"
            try:
                target_path.write_bytes(file_bytes)
                st.sidebar.success(f"Datei erfolgreich im Ordner gespeichert: {target_path.name}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Fehler beim Speichern der Datei: {e}")
                
    if st.button("🗑️ Gespeicherte Historie löschen"):
        if is_using_gsheets():
            save_store_hybrid(spreadsheet_name, pd.DataFrame(columns=CANONICAL_COLUMNS))
        else:
            if STORE_PATH.exists():
                STORE_PATH.unlink()
            load_store_hybrid.clear()
        for f in BROKER_CSV_DIR.glob("*.csv"):
            f.unlink()
        st.rerun()

# ---------------------------------------------------------------------------
# HYBRID-DATEN ABFRAGEN
# ---------------------------------------------------------------------------

tx, num_scanned = process_broker_csv_folder_hybrid(spreadsheet_name)
tx = pre_process_corporate_actions(tx)

if num_scanned > 0:
    st.sidebar.info(f"ℹ️ {num_scanned} neue Transaktionen wurden erfolgreich importiert!")

# NEU: Die Definition direkt nach dem Laden von 'tx' platzieren
ignored_tx_ids, inbound_to_outbound = identify_portfolio_transfers(tx)

cash_data = load_cash_values_hybrid(spreadsheet_name)

# ---------------------------------------------------------------------------
# SIDEBAR 2: WEITERE KONTEN RENDERING (NUR SPEICHERN BEI KLICK)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.header("2. Weitere Konten") # Auf 2 geändert
    st.caption("Trage Cash-Bestände, Gold, Silber und Verbindlichkeiten ein.")
    
    with st.form(f"cash_form_{spreadsheet_name}", clear_on_submit=False):
        input_tagesgeld = st.number_input("Tagesgeld (€)", value=float(cash_data.get("tagesgeld", 0.0)), step=100.0, format="%.2f", key=f"tagesgeld_{spreadsheet_name}")
        input_girokonto = st.number_input("Girokonto (€)", value=float(cash_data.get("girokonto", 0.0)), step=100.0, format="%.2f", key=f"girokonto_{spreadsheet_name}")
        
        # NEU: Datums-Wähler für das Cash-Guthaben
        input_cash_date = st.date_input("Tagesgeld-/Cash-Datum", value=pd.to_datetime(cash_data.get("cash_date", "2021-01-01")), key=f"cash_date_sel_{spreadsheet_name}")
        
        input_andere_assets = st.number_input("Andere Assets (Immobilien, geschätzt) (€)", value=float(cash_data.get("andere_assets", 0.0)), step=1000.0, format="%.2f", key=f"andere_assets_{spreadsheet_name}")
        input_darlehen = st.number_input("Offenes Darlehen / Kredite (€)", value=float(cash_data.get("darlehen", 0.0)), step=1000.0, format="%.2f", key=f"darlehen_{spreadsheet_name}")
        
        st.markdown("**Gold-Bestand**")
        input_gold_unit = st.selectbox("Einheit (Gold)", ["Unzen (oz)", "Gramm (g)"], index=0 if cash_data.get("gold_unit") == "Unzen (oz)" else 1, key=f"gold_unit_sel_{spreadsheet_name}")
        input_gold_amount = st.number_input("Gold-Menge", value=float(cash_data.get("gold_amount", 0.0)), step=0.1, format="%.4f", key=f"gold_amt_sel_{spreadsheet_name}")
        input_gold_cost = st.number_input("Gold-Kaufpreis gesamt (€)", value=float(cash_data.get("gold_cost", 0.0)), step=100.0, format="%.2f", key=f"gold_cost_sel_{spreadsheet_name}")
        input_gold_date = st.date_input("Gold-Kaufdatum", value=pd.to_datetime(cash_data.get("gold_date", "2021-01-01")), key=f"gold_date_sel_{spreadsheet_name}")
        
        st.markdown("**Silber-Bestand**")
        input_silver_unit = st.selectbox("Einheit (Silber)", ["Unzen (oz)", "Gramm (g)"], index=0 if cash_data.get("silver_unit") == "Unzen (oz)" else 1, key=f"silver_unit_sel_{spreadsheet_name}")
        input_silver_amount = st.number_input("Silber-Menge", value=float(cash_data.get("silver_amount", 0.0)), step=1.0, format="%.4f", key=f"silver_amt_sel_{spreadsheet_name}")
        input_silver_cost = st.number_input("Silber-Kaufpreis gesamt (€) inkl. MwSt", value=float(cash_data.get("silver_cost", 0.0)), step=100.0, format="%.2f", key=f"silver_cost_sel_{spreadsheet_name}")
        input_silver_date = st.date_input("Silber-Kaufdatum", value=pd.to_datetime(cash_data.get("silver_date", "2021-01-01")), key=f"silver_date_sel_{spreadsheet_name}")
        
        gold_date_str_input = input_gold_date.strftime("%Y-%m-%d")
        silver_date_str_input = input_silver_date.strftime("%Y-%m-%d")
        cash_date_str_input = input_cash_date.strftime("%Y-%m-%d")  # NEU
        
        st.caption("💡 Hinweis: Offene Kredite bitte als *positive* Zahl eintragen. Sie werden im Gesamtvermögen automatisch abgezogen.")
        
        submit_button = st.form_submit_button("💾 Änderungen speichern", use_container_width=True)
        
        if submit_button:
            save_cash_values_hybrid(
                spreadsheet_name,
                input_tagesgeld,
                input_girokonto,
                input_darlehen,
                input_andere_assets,
                cash_date_str_input,  # NEU: Übergibt das gewählte Datum beim Speichern
                input_gold_unit,
                input_gold_amount,
                input_gold_cost,
                gold_date_str_input,
                input_silver_unit,
                input_silver_amount,
                input_silver_cost,
                silver_date_str_input
            )
            st.success("Werte erfolgreich gespeichert!")
            st.rerun()
# ---------------------------------------------------------------------------
# WEITERE BERECHNUNGSWERTE ABLEITEN
# ---------------------------------------------------------------------------

total_cash = input_tagesgeld + input_girokonto
total_other_assets = input_andere_assets
total_loans = input_darlehen
cash_date_str = cash_date_str_input  # NEU

gold_ounces = input_gold_amount / OZ_TO_G if input_gold_unit == "Gramm (g)" else input_gold_amount
gold_cost = input_gold_cost
gold_date_str = gold_date_str_input

silver_ounces = input_silver_amount / OZ_TO_G if input_silver_unit == "Gramm (g)" else input_silver_amount
silver_cost = input_silver_cost
silver_date_str = silver_date_str_input

# Virtual Transactions für Gold, Silber, Cash, Andere Assets & Kredite in tx einbetten
tx = tx.copy()
virtual_tx_list = []
if gold_ounces > 0:
    virtual_tx_list.append({
        "date": pd.to_datetime(gold_date_str),
        "ISIN": "Physisches Gold",
        "Name": "Gold (Physisch, 99.9%)",
        "type": "BUY",
        "asset_class": "STOCK",
        "shares": gold_ounces,
        "price": gold_cost / gold_ounces,
        "amount": -gold_cost,
        "fee": 0.0,
        "tx_id": "virtual_gold_buy"
    })
if silver_ounces > 0:
    virtual_tx_list.append({
        "date": pd.to_datetime(silver_date_str),
        "ISIN": "Physisches Silber",
        "Name": "Silber (Physisch)",
        "type": "BUY",
        "asset_class": "STOCK",
        "shares": silver_ounces,
        "price": silver_cost / silver_ounces,
        "amount": -silver_cost,
        "fee": 0.0,
        "tx_id": "virtual_silver_buy"
    })
if total_cash > 0:
    virtual_tx_list.append({
        "date": pd.to_datetime(cash_date_str),  
        "ISIN": "Physisches Cash",
        "Name": "Cash (Giro/Tagesgeld)",
        "type": "BUY",
        "asset_class": "CASH",
        "shares": total_cash,
        "price": 1.0,
        "amount": -total_cash,
        "fee": 0.0,
        "tx_id": "virtual_cash_buy"
    })
if total_other_assets > 0:
    virtual_tx_list.append({
        "date": pd.Timestamp.now().normalize(),
        "ISIN": "Andere Assets",
        "Name": "Andere Assets (Sachwerte)",
        "type": "BUY",
        "asset_class": "OTHER",
        "shares": total_other_assets,
        "price": 1.0,
        "amount": -total_other_assets,
        "fee": 0.0,
        "tx_id": "virtual_other_assets_buy"
    })
if total_loans > 0:
    virtual_tx_list.append({
        "date": pd.Timestamp.now().normalize(),
        "ISIN": "Offene Kredite",
        "Name": "Offene Kredite / Darlehen",
        "type": "BUY",
        "asset_class": "LIABILITY",
        "shares": -total_loans,
        "price": 1.0,
        "amount": total_loans,
        "fee": 0.0,
        "tx_id": "virtual_loans_buy"
    })

if virtual_tx_list:
    tx = pd.concat([tx, pd.DataFrame(virtual_tx_list)], ignore_index=True)

# NEU: Wenn tx immer noch komplett leer ist, stoppe die App sauber 
# und zeige ein Willkommens-Fenster zur Anleitung
if tx.empty:
    st.info(
        "👋 **Willkommen in deiner neuen Portfolio-Analyse!**\n\n"
        "Dein Online-Speicher ist aktuell noch komplett leer. Um zu starten, führe einfach einen der folgenden Schritte aus:\n\n"
        "1. **Broker-Daten importieren:** Lade deine ersten Broker-Exporte (CSV von Scalable oder Trade Republic) links in der Sidebar unter **1. Portfolio-Daten** hoch.\n"
        "2. **Kontostand eintragen:** Trage dein erstes Tagesgeld oder Guthaben links in der Sidebar unter **3. Weitere Konten** ein und klicke auf **Änderungen speichern**.\n\n"
        "Sobald du Daten eingetragen hast, baut sich dein Cockpit vollautomatisch auf!"
    )
    st.stop()

# Summe aller Dividenden aus den Transaktionen berechnen
total_dividends = tx[tx["type"] == "DIVIDEND"]["amount"].sum() if not tx.empty else 0.0

# --- Ticker-Zuordnung im Hintergrund laden (ohne Anzeige in der Sidebar) ---
all_isins = sorted(tx["ISIN"].unique())
VIRTUAL_ISINS = ["Physisches Gold", "Physisches Silber", "Physisches Cash", "Andere Assets", "Offene Kredite"]
ticker_input_isins = [isin for isin in all_isins if isin not in VIRTUAL_ISINS]
isin_names = tx.groupby("ISIN")["Name"].last().to_dict()
ticker_overrides = load_ticker_overrides(ticker_input_isins)

# Zuordnung still im Hintergrund aufbauen, um die Sidebar clean zu halten
ticker_map = {}
for isin in ticker_input_isins:
    val_override = ticker_overrides.get(isin, "")
    default_val = val_override if val_override != "" else DEFAULT_ISIN_TICKER_MAP.get(isin, "")
    ticker_map[isin] = default_val.strip() if default_val else ""

# ---------------------------------------------------------------------------
# POSITIONSVERARBEITUNG & LIVE-KURSE
# ---------------------------------------------------------------------------
with st.spinner("Verarbeite Positionen..."):
    open_df, closed_df, unresolved_free_receipts = build_positions(tx, ticker_map)

if open_df.empty:
    tickers = ("XAUUSD=X", "SI=F")
    last_update_time = pd.Timestamp.now().strftime("%d.%m.%Y %H:%M:%S")
    current_prices = {"XAUUSD=X": 0.0, "SI=F": 0.0}
else:
    tickers_set = {t for t in ticker_map.values() if t}
    tickers_set.add("XAUUSD=X")  # Live-Spot-Goldpreis mit anfragen
    tickers_set.add("SI=F")  # Live-Silberpreis mit anfragen
    tickers = tuple(sorted(tickers_set))
    with st.spinner("Lade Live-Kurse..."):
        current_prices, last_update_time = fetch_current_prices(tickers)

# Typ der Ticker-Spalte auf Objekt setzen, um Pandas Assignment-Fehler bei leeren DataFrames zu verhindern
open_df["Ticker"] = open_df["ISIN"].map(ticker_map).astype(object)

open_df.loc[open_df["ISIN"] == "Physisches Gold", "Ticker"] = "XAUUSD=X"  # Auf Spot-Gold geändert
open_df.loc[open_df["ISIN"] == "Physisches Silber", "Ticker"] = "SI=F"

# ---------------------------------------------------------------------------
# WEITERE BERECHNUNGSWERTE ABLEITEN & ABSICHERN
# ---------------------------------------------------------------------------

total_cash = input_tagesgeld + input_girokonto
total_other_assets = input_andere_assets
total_loans = input_darlehen
cash_date_str = cash_date_str_input  

gold_ounces_raw = input_gold_amount / OZ_TO_G if input_gold_unit == "Gramm (g)" else input_gold_amount
val_gold_ounces = float(gold_ounces_raw) if (gold_ounces_raw is not None and pd.notna(gold_ounces_raw)) else 0.0
gold_cost = input_gold_cost
gold_date_str = gold_date_str_input

silver_ounces_raw = input_silver_amount / OZ_TO_G if input_silver_unit == "Gramm (g)" else input_silver_amount
val_silver_ounces = float(silver_ounces_raw) if (silver_ounces_raw is not None and pd.notna(silver_ounces_raw)) else 0.0
silver_cost = input_silver_cost
silver_date_str = silver_date_str_input

# Live-Rohstoffwerte in EUR bestimmen (sicher gegen temporäre Ladefehler abgesichert)
val_gold_price = current_prices.get("XAUUSD=X")
live_gold_price = float(val_gold_price) if (val_gold_price is not None and pd.notna(val_gold_price)) else 0.0
total_gold_value = val_gold_ounces * live_gold_price

val_silver_price = current_prices.get("SI=F")
live_silver_price = float(val_silver_price) if (val_silver_price is not None and pd.notna(val_silver_price)) else 0.0
total_silver_value = val_silver_ounces * live_silver_price

open_df["Aktueller Kurs"] = open_df["Ticker"].map(current_prices)
open_df.loc[open_df["ISIN"] == "Physisches Gold", "Aktueller Kurs"] = live_gold_price
open_df.loc[open_df["ISIN"] == "Physisches Silber", "Aktueller Kurs"] = live_silver_price

open_df.loc[open_df["ISIN"] == "Physisches Cash", "Aktueller Kurs"] = 1.0
open_df.loc[open_df["ISIN"] == "Andere Assets", "Aktueller Kurs"] = 1.0
open_df.loc[open_df["ISIN"] == "Offene Kredite", "Aktueller Kurs"] = 1.0

open_df.loc[open_df["ISIN"] == "Physisches Cash", "invested"] = total_cash
open_df.loc[open_df["ISIN"] == "Physisches Cash", "avg_cost"] = 1.0

open_df.loc[open_df["ISIN"] == "Andere Assets", "invested"] = total_other_assets
open_df.loc[open_df["ISIN"] == "Andere Assets", "avg_cost"] = 1.0

open_df.loc[open_df["ISIN"] == "Offene Kredite", "invested"] = -total_loans
open_df.loc[open_df["ISIN"] == "Offene Kredite", "avg_cost"] = 1.0

# WICHTIG: Wandelt alle eventuellen None/NaN-Kurse in 0.0 um, um Multiplikations-Abstürze dauerhaft zu verhindern!
open_df["Aktueller Kurs"] = pd.to_numeric(open_df["Aktueller Kurs"], errors="coerce").fillna(0.0)

# ---------------------------------------------------------------------------
# KORREKTUR: DERIVATE FILTERN
# ---------------------------------------------------------------------------
derivatives_mask = (
    (open_df["Name"].str.contains("Long|Short|Put|Call|Mini|Turbo|Faktor", case=False, na=False)) | 
    (open_df["asset_class"].str.lower() == "derivative") |
    (open_df["ISIN"].str.startswith("DE000SJ")) | 
    (open_df["ISIN"].str.startswith("DE000VG")) |
    (open_df["ISIN"].str.startswith("DE000HT")) |
    (open_df["ISIN"].str.startswith("DE000UJ"))
) & (open_df["Aktueller Kurs"].isna())

expired_derivatives = open_df[derivatives_mask].copy()
if not expired_derivatives.empty:
    open_df = open_df[~derivatives_mask].copy()
    
    for _, row in expired_derivatives.iterrows():
        new_closed_row = {
            "ISIN": row["ISIN"],
            "Name": row["Name"],
            "asset_class": row["asset_class"],
            "total_invested": row["invested"],
            "total_proceeds": 0.0,
            "realized_pl": -row["invested"],
            "realized_pl_pct": -100.0,
            "first_date": row["first_date"],
            "last_date": pd.Timestamp.now(),
        }
        closed_df = pd.concat([closed_df, pd.DataFrame([new_closed_row])], ignore_index=True)

open_df["Aktueller Wert"] = open_df["shares"] * open_df["Aktueller Kurs"]
open_df["Gewinn/Verlust (€)"] = open_df["Aktueller Wert"] - open_df["invested"]
open_df["Gewinn/Verlust (%)"] = open_df["Gewinn/Verlust (€)"] / open_df["invested"].replace(0, np.nan) * 100

# ---------------------------------------------------------------------------
# SORTIERUNG ABSTEIGEND
# ---------------------------------------------------------------------------
open_df = open_df.sort_values("invested", ascending=False).reset_index(drop=True)

missing = open_df[open_df["Aktueller Kurs"].isna()]
if not missing.empty:
    st.warning(f"Ticker in der Sidebar prüfen für: {', '.join(missing['ISIN'].tolist())}")

# Berechnungen für Kennzahlen
total_invested = open_df["invested"].sum()
total_value = open_df["Aktueller Wert"].sum(skipna=True)
total_pl_eur = total_value - total_invested
total_pl_pct = (total_pl_eur / total_invested * 100) if total_invested else 0
total_realized = closed_df["realized_pl"].sum() if not closed_df.empty else 0.0
total_fees = tx["fee"].fillna(0).abs().sum() # Bildet erst den Absolutwert jeder Zeile und addiert sie dann auf
# NEU: Diese beiden Berechnungen direkt hierhin verschieben
total_return_abs = total_pl_eur + total_realized + total_dividends
total_return_pct = (total_return_abs / total_invested * 100) if total_invested else 0.0

# Separiere reinen Wertpapierwert ohne Edelmetalle und Cash/andere Assets/Kredite für Snapshot-Kacheln
VIRTUAL_ISINS = ["Physisches Gold", "Physisches Silber", "Physisches Cash", "Andere Assets", "Offene Kredite"]
total_value_securities = open_df[~open_df["ISIN"].isin(VIRTUAL_ISINS)]["Aktueller Wert"].sum(skipna=True)
net_worth = total_value

# --- HIER HISTORISCHEN WERTE-VERLAUF VORAB LADEN ---
earliest_date = tx["date"].min().strftime("%Y-%m-%d")
with st.spinner("Lade Kurshistorie..."):
    price_history = fetch_price_history(tickers, earliest_date)
value_history = build_portfolio_value_history(tx, price_history, ticker_map)

if not value_history.empty:
    if "GC=F" in price_history.columns:
        gold_start_dt = pd.to_datetime(gold_date_str)
        gold_mask = value_history.index >= gold_start_dt
        if gold_mask.any():
            value_history.loc[gold_mask, "Portfolio-Wert"] += gold_ounces * price_history.loc[gold_mask, "GC=F"]
            value_history.loc[gold_mask, "Eingezahltes Kapital"] += gold_cost
            
    if "SI=F" in price_history.columns:
        silver_start_dt = pd.to_datetime(silver_date_str)
        silver_mask = value_history.index >= silver_start_dt
        if silver_mask.any():
            value_history.loc[silver_mask, "Portfolio-Wert"] += silver_ounces * price_history.loc[silver_mask, "SI=F"]
            value_history.loc[silver_mask, "Eingezahltes Kapital"] += silver_cost

    value_history["Portfolio-Wert"] += total_cash + total_other_assets - total_loans
    value_history["Eingezahltes Kapital"] += total_cash + total_other_assets - total_loans

# XIRR / IZF Cashflow-Berechnung
cash_flows = []
tx_cf_active = tx[~tx["tx_id"].isin(ignored_tx_ids)].copy()
for _, row in tx_cf_active.iterrows():
    if row["ISIN"] in ["Physisches Cash", "Andere Assets", "Offene Kredite"]:
        continue
    if row["type"] not in ["TRANSFER", "FREE_RECEIPT"]:
        cash_flows.append((row["date"], row["amount"]))
if total_value > 0:
    measured_final_value = total_value - total_cash - total_other_assets - (-total_loans)
    cash_flows.append((pd.Timestamp.now(), measured_final_value))
izf_val = xirr(cash_flows)

# -----------------------------------------------------------------
# SCHNAPPSCHUSS & PERFORMANCE-RASTER (TOP)
# -----------------------------------------------------------------
st.markdown("---")

st.subheader("📊 Schnappschuss (Gesamtübersicht)")
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6, col_m7 = st.columns(7)
col_m1.metric("Gesamtvermögen", f"{net_worth:,.2f} €")
col_m2.metric("Depot (Wertpapiere)", f"{total_value_securities:,.2f} €")
col_m3.metric("Cash (Giro/Tagesgeld)", f"{total_cash:,.2f} €")
col_m4.metric("Gold-Bestand", f"{total_gold_value:,.2f} €")
col_m5.metric("Silber-Bestand", f"{total_silver_value:,.2f} €")
col_m6.metric("Andere Assets", f"{total_other_assets:,.2f} €")
col_m7.metric(
    "Offene Kredite", 
    f"-{total_loans:,.2f} €" if total_loans > 0 else "0.00 €", 
    delta=f"-{total_loans:,.2f} €" if total_loans > 0 else None, 
    delta_color="inverse"
)

st.markdown("<br>", unsafe_allow_html=True)

perf_header_col1, perf_header_col2, perf_header_col3 = st.columns([3.5, 2.5, 1])

with perf_header_col1:
    st.markdown("##### 📈 Depot-Performance über verschiedene Zeiträume")
    
with perf_header_col2:
    st.markdown(
        f"<div style='text-align: right; font-size: 14px; color: rgba(255,255,255,0.85); padding-top: 8px; font-weight: 500;'>"
        f"Live-Kurse zuletzt aktualisiert: {last_update_time}"
        f"</div>", 
        unsafe_allow_html=True
    )
    
with perf_header_col3:
    if st.button("🔄 Aktualisieren", use_container_width=True):
        fetch_current_prices.clear()
        fetch_price_history.clear()
        st.rerun()

if not value_history.empty:
    today_date = value_history.index[-1]
    periods = {
        "Heute (1T)": today_date - pd.Timedelta(days=1),
        "1 Woche (7T)": today_date - pd.Timedelta(days=7),
        "1 Monat (30T)": today_date - pd.Timedelta(days=30),
        "YTD (Jahr)": pd.Timestamp(year=today_date.year - 1, month=12, day=31),
        "1 Jahr (365T)": today_date - pd.Timedelta(days=365),
        "MAX (Gesamt)": None  # Sonderbehandlung für die reale Gesamt-Performance
    }
    
    p_cols = st.columns(6)
    for i, (label, target_dt) in enumerate(periods.items()):
        with p_cols[i]:
            if label == "MAX (Gesamt)":
                # Hier weisen wir exakt die reale, korrekte Performance aus der Tabelle aus
                sign = "+" if total_return_abs >= 0 else ""
                st.metric(
                    label=label,
                    value=f"{total_return_abs:,.2f} €",
                    delta=f"{sign}{total_return_pct:.2f}%"
                )
            else:
                perf = get_period_performance(value_history, target_dt)
                if perf is not None:
                    pl_eur, pl_pct = perf
                    sign = "+" if pl_eur >= 0 else ""
                    st.metric(
                        label=label,
                        value=f"{pl_eur:,.2f} €",
                        delta=f"{sign}{pl_pct:.2f}%"
                    )
                else:
                    st.metric(label=label, value="-", delta=None)
else:
    st.info("Kurshistorie wird geladen, um Performance-Indikatoren anzuzeigen.")

st.markdown("---")

# Kennzahlen-Tabelle (Copilot-Style)
st.subheader("📋 Kennzahlen (Copilot-Style)")

total_invested_securities = open_df[~open_df["ISIN"].isin(VIRTUAL_ISINS)]["invested"].sum()
total_unrealized_pl_securities = total_value_securities - total_invested_securities
total_unrealized_pct_securities = (total_unrealized_pl_securities / total_invested_securities * 100) if total_invested_securities else 0.0

# Formatierungen für Kennzahlen (Verhindert farbige Nullen und doppelte Einfärbung)
sign_unrealized_sec = "↑" if total_unrealized_pl_securities > 1e-4 else ("↓" if total_unrealized_pl_securities < -1e-4 else "")
color_unrealized_sec = "#2ca02c" if total_unrealized_pl_securities > 1e-4 else ("#d62728" if total_unrealized_pl_securities < -1e-4 else "inherit")

sign_realized = "↑" if total_realized > 1e-4 else ("↓" if total_realized < -1e-4 else "")
color_realized = "#2ca02c" if total_realized > 1e-4 else ("#d62728" if total_realized < -1e-4 else "inherit")

# Gold Live Performance
gold_pl_eur = total_gold_value - gold_cost
gold_pl_pct = (gold_pl_eur / gold_cost * 100) if gold_cost > 0 else 0.0
sign_gold = "↑" if gold_pl_eur > 1e-4 else ("↓" if gold_pl_eur < -1e-4 else "")
color_gold = "#2ca02c" if gold_pl_eur > 1e-4 else ("#d62728" if gold_pl_eur < -1e-4 else "inherit")

# Silber Live Performance
silver_pl_eur = total_silver_value - silver_cost
silver_pl_pct = (silver_pl_eur / silver_cost * 100) if silver_cost > 0 else 0.0
sign_silver = "↑" if silver_pl_eur > 1e-4 else ("↓" if silver_pl_eur < -1e-4 else "")
color_silver = "#2ca02c" if silver_pl_eur > 1e-4 else ("#d62728" if silver_pl_eur < -1e-4 else "inherit")

sign_total = "↑" if total_return_abs > 1e-4 else ("↓" if total_return_abs < -1e-4 else "")
color_total = "#2ca02c" if total_return_abs > 1e-4 else ("#d62728" if total_return_abs < -1e-4 else "inherit")

# Dynamische Steuerung für Kredite, Dividenden & Gebühren bei Nullwerten
color_loans = "#d62728" if total_loans > 1e-4 else "inherit"
sign_loans = "-" if total_loans > 1e-4 else ""

color_dividends = "#2ca02c" if total_dividends > 1e-4 else "inherit"
sign_dividends = "↑ " if total_dividends > 1e-4 else ""

color_fees = "#d62728" if total_fees > 1e-4 else "inherit"
sign_fees = "↓ -" if total_fees > 1e-4 else ""

izf_str = "-"
if izf_val is not None:
    izf_pct = izf_val * 100.0
    sign_izf = "↑" if izf_pct >= 0 else "↓"
    color_izf = "#2ca02c" if izf_pct >= 0 else "#d62728"
    izf_str = f"<span style='color: {color_izf}; font-weight: bold;'>{sign_izf} {izf_pct:.2f}%</span>"

st.markdown(
    f"""
    <div style="background-color: rgba(28, 40, 65, 0.4); padding: 22px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1); font-family: sans-serif;">
        <table style="width: 100%; border-collapse: collapse; font-size: 15px; color: inherit;">
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong style="font-size: 15px;">Investiertes Kapital (Wertpapiere)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Das aktuell in deinen offenen Depotpositionen gebundene Einstandskapital (ohne Edelmetalle).">ℹ️</span>
                </td>
                <td style="text-align: right; font-weight: bold; font-size: 15px;">{total_invested_securities:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Kursgewinn (unrealisiert, Wertpapiere)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Der noch nicht realisierte Wertzuwachs deiner aktuellen Bestände (ohne Edelmetalle) im Vergleich zu deinen Kaufkosten.">ℹ️</span>
                </td>
                <td style="text-align: right;">
                    <span style="color: {color_unrealized_sec}; font-weight: bold; margin-right: 15px;">{sign_unrealized_sec} {total_unrealized_pct_securities:.2f}%</span>
                    <span style="color: {color_unrealized_sec}; font-weight: bold;">{total_unrealized_pl_securities:,.2f} €</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Realisierte Gewinne</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Bereits durch Verkäufe realisierte Gewinne und Verluste über die gesamte Historie hinweg (nach der Durchschnittskostenmethode).">ℹ️</span>
                </td>
                <td style="text-align: right;">
                    <span style="color: {color_realized}; font-weight: bold;">{sign_realized} {total_realized:,.2f} €</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Gebühren / Transaktionskosten / Steuern</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Summe aller in der Historie erfassten Transaktionsgebühren.">ℹ️</span>
                </td>
                <td style="text-align: right; color: {color_fees}; font-weight: bold;">{sign_fees}{total_fees:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Cash-Bestand (Tagesgeld & Giro)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Die Summe deiner manuell in der Sidebar hinterlegten Guthaben auf Giro- und Tagesgeldkonten.">ℹ️</span>
                </td>
                <td style="text-align: right; font-weight: bold; font-size: 15px;">{total_cash:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Gold-Bestand (Live-Wert)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Der aktuelle Wert deines physischen Goldes basierend auf dem Weltmarkt-Livepreis (Kaufpreis: {gold_cost:,.2f} €).">ℹ️</span>
                </td>
                <td style="text-align: right;">
                    <span style="color: {color_gold}; font-weight: bold; margin-right: 15px;">{sign_gold} {gold_pl_pct:+.2f}%</span>
                    <span style="font-weight: bold;">{total_gold_value:,.2f} €</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Silber-Bestand (Live-Wert)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Der aktuelle Wert deines physischen Silbers basierend auf dem Weltmarkt-Livepreis, inklusive 19% MwSt. (Kaufpreis: {silver_cost:,.2f} €).">ℹ️</span>
                </td>
                <td style="text-align: right;">
                    <span style="color: {color_silver}; font-weight: bold; margin-right: 15px;">{sign_silver} {silver_pl_pct:+.2f}%</span>
                    <span style="font-weight: bold;">{total_silver_value:,.2f} €</span>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Andere Assets (Immobilien, Sachwerte)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Manuell eingetragener Schätzwert von Immobilien oder anderen physischen Sachwerten.">ℹ️</span>
                </td>
                <td style="text-align: right; font-weight: bold; font-size: 15px;">{total_other_assets:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Offene Kredite / Darlehen</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Verbindlichkeiten und Schulden (werden vom Gesamtvermögen abgezogen).">ℹ️</span>
                </td>
                <td style="text-align: right; color: {color_loans}; font-weight: bold; font-size: 15px;">{sign_loans}{total_loans:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Dividenden & Ausschüttungen</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Die Summe aller erhaltenen Dividendenauszahlungen.">ℹ️</span>
                </td>
                <td style="text-align: right; color: {color_dividends}; font-weight: bold;">{sign_dividends}{total_dividends:,.2f} €</td>
            </tr>
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); height: 45px;">
                <td>
                    <strong>Gesamt-Performance (Depot)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Der absolute Gesamtertrag deines Depots (unrealisierte Kursgewinne + realisierte Gewinne + erhaltene Dividenden).">ℹ️</span>
                </td>
                <td style="text-align: right;">
                    <span style="color: {color_total}; font-weight: bold; font-size: 15px; margin-right: 15px;">{sign_total} {total_return_pct:.2f}%</span>
                    <span style="color: {color_total}; font-weight: bold; font-size: 15px;">{total_return_abs:,.2f} €</span>
                </td>
            </tr>
            <tr style="border-bottom: 2px solid rgba(255, 255, 255, 0.2); height: 50px;">
                <td>
                    <strong>Gesamtvermögen</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Dein aggregierter Vermögenswert aus dem Depotwert, den Cash-Beständen, Gold und Sachwerten abzüglich der Kredite.">ℹ️</span>
                </td>
                <td style="text-align: right; font-weight: bold; font-size: 16px;">{net_worth:,.2f} €</td>
            </tr>
            <tr style="height: 50px;">
                <td>
                    <strong>Performance p.a. (IZF / XIRR)</strong> 
                    <span style="cursor: help; margin-left: 5px;" title="Die annualisierte zeitgewichtete Rendite unter Berücksichtigung aller Zu- und Abflüsse sowie dem heutigen Depotwert (bezogen auf investierte Vermögenswerte).">ℹ️</span>
                </td>
                <td style="text-align: right; font-size: 15px; color: {color_total}; font-weight: bold;">{izf_str}</td>
            </tr>
        </table>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# POSITIONSTABELLE
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Positionsübersicht")

# Dollar-Umrechnungskurs live anzeigen
usd_eur_rate = fetch_usd_eur_rate()
st.caption(f"💵 Aktueller Wechselkurs: **1 USD = {usd_eur_rate:.4f} EUR** (Alle US-Dollar-Wertpapiere und Rohstoffe werden live umgerechnet)")

display_cols = {
    "Name": "Name",
    "ISIN": "ISIN",
    "Ticker": "Ticker",
    "shares": "Anteile",
    "avg_cost": "Ø Kaufkurs (€)",
    "invested": "Investiert (€)",
    "Aktueller Kurs": "Aktueller Kurs (€)",
    "Aktueller Wert": "Aktueller Wert (€)",
    "Gewinn/Verlust (€)": "G/V (€)",
    "Gewinn/Verlust (%)": "G/V (%)",
}
table = open_df.rename(columns=display_cols)[list(display_cols.values())]
st.dataframe(
    table.style.format(
        {
            "Anteile": "{:.4f}",
            "Ø Kaufkurs (€)": "{:.2f}",
            "Investiert (€)": "{:,.2f}",
            "Aktueller Kurs (€)": "{:.2f}",
            "Aktueller Wert (€)": "{:,.2f}",
            "G/V (€)": "{:,.2f}",
            "G/V (%)": "{:.2f}",
        }
    ).map(color_profit_loss, subset=["G/V (%)"]), # Ersetzt background_gradient durch die neue Farbfunktion
    width="stretch",
)
# ---------------------------------------------------------------------------
# CHARTS: PIE + VERLAUF UNTEREINANDER
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Portfolio-Verteilung")
pie_df = open_df.dropna(subset=["Aktueller Wert"])
if not pie_df.empty:
    fig_pie = px.pie(pie_df, names="Name", values="Aktueller Wert", hole=0.45)
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    
    # Legende nach unten schieben
    fig_pie.update_layout(legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", yanchor="top"))
    
    st.plotly_chart(fig_pie, width="stretch")

st.markdown("---")
st.subheader("Wertverlauf über Zeit & Transaktionen")
if not value_history.empty:
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=value_history.index, y=value_history["Portfolio-Wert"], name="Gesamtvermögen", line=dict(width=2)))
    fig_line.add_trace(go.Scatter(x=value_history.index, y=value_history["Eingezahltes Kapital"], name="Eingezahltes Kapital (inkl. Cash/Metalle)", line=dict(width=2, dash="dash")))
    
    tx_active = tx[~tx["tx_id"].isin(ignored_tx_ids)].copy()
    tx_in_range = tx_active[(tx_active["date"] >= value_history.index.min()) & (tx_active["date"] <= value_history.index.max())].copy()
    
    if not tx_in_range.empty:
        m_y = []
        for d in tx_in_range["date"]:
            idx_loc = value_history.index.get_indexer([d], method="nearest")[0]
            m_y.append(value_history.iloc[idx_loc]["Portfolio-Wert"])
        tx_in_range["y_val"] = m_y

        hover_texts = []
        for _, row in tx_in_range.iterrows():
            action_label = "Kauf" if row["type"] == "BUY" else "Verkauf" if row["type"] == "SELL" else "Einbuchung/Zuteilung"
            broker_label = get_broker_name(row["tx_id"])
            hover_texts.append(
                f"<b>{row['Name']}</b> ({row['ISIN']})<br>"
                f"Bezug: {broker_label}<br>"
                f"Datum: {row['date'].strftime('%d.%m.%Y')}<br>"
                f"Aktion: {action_label}<br>"
                f"Menge/Anteile: {abs(row['shares']):.4f}<br>"
                f"Kurs: {row['price']:.2f} €<br>"
                f"Wert: {abs(row['amount']):,.2f} €"
            )
        tx_in_range["hover_text"] = hover_texts

        buys = tx_in_range[tx_in_range["type"] == "BUY"]
        if not buys.empty:
            fig_line.add_trace(go.Scatter(
                x=buys["date"], y=buys["y_val"], mode="markers", name="Kauf (BUY)",
                marker=dict(symbol="triangle-up", size=9, color="#2ca02c", line=dict(width=1, color="black")),
                text=buys["hover_text"], hoverinfo="text"
            ))

        sells = tx_in_range[tx_in_range["type"] == "SELL"]
        if not sells.empty:
            fig_line.add_trace(go.Scatter(
                x=sells["date"], y=sells["y_val"], mode="markers", name="Verkauf (SELL)",
                marker=dict(symbol="triangle-down", size=9, color="#d62728", line=dict(width=1, color="black")),
                text=sells["hover_text"], hoverinfo="text"
            ))

        receipts = tx_in_range[tx_in_range["type"].isin(["FREE_RECEIPT", "TRANSFER", "SPLIT"])]
        if not receipts.empty:
            fig_line.add_trace(go.Scatter(
                x=receipts["date"], y=receipts["y_val"], mode="markers", name="Zuteilung/Transfer",
                marker=dict(symbol="star", size=10, color="#1f77b4", line=dict(width=1, color="black")),
                text=receipts["hover_text"], hoverinfo="text"
            ))

    # --- Depotübertragungen ---
    ignored_tx_ids, inbound_to_outbound = identify_portfolio_transfers(tx)
    transfers_tx = tx[tx["tx_id"].isin(inbound_to_outbound.keys())].copy()
    
    if not transfers_tx.empty and not value_history.empty:
        transfer_dates_df = transfers_tx.groupby("date").first().reset_index()
        
        for _, row_date in transfer_dates_df.iterrows():
            date_val = row_date["date"]
            fig_line.add_shape(
                type="line",
                x0=date_val,
                y0=0,
                x1=date_val,
                y1=1,
                yref="paper",
                line=dict(color="orange", width=1.2, dash="dash"),
                layer="below"
            )

        m_y = []
        for _, row_date in transfer_dates_df.iterrows():
            d = row_date["date"]
            idx_loc = value_history.index.get_indexer([d], method="nearest")[0]
            m_y.append(value_history.iloc[idx_loc]["Portfolio-Wert"])
        transfer_dates_df["y_val"] = m_y

        hover_texts = []
        for _, row_grp in transfer_dates_df.iterrows():
            date_val = row_grp["date"]
            day_transfers = transfers_tx[transfers_tx["date"] == date_val]
            detail_lines = []
            for _, row in day_transfers.iterrows():
                in_id = row["tx_id"]
                out_id = inbound_to_outbound.get(in_id)
                
                dest_broker = get_broker_name(in_id)
                if out_id == "unmatched":
                    src_broker = "Anderer/unbekannter Broker"
                elif out_id:
                    src_broker = get_broker_name(out_id)
                else:
                    src_broker = "Unbekannt"
                    
                detail_lines.append(
                    f"• <b>{row['Name']}</b> ({abs(row['shares']):.4f} Anteile)<br>"
                    f"  Transfer: Von <i>{src_broker}</i> zu <i>{dest_broker}</i>"
                )
            
            tooltip_text = (
                f"<b>Depotübertrag (Eingegangen)</b><br>"
                f"Datum: {date_val.strftime('%d.%m.%Y')}<br><br>"
                f"Übertragene Positionen:<br>" + "<br>".join(detail_lines)
            )
            hover_texts.append(tooltip_text)
        
        transfer_dates_df["hover_text"] = hover_texts

        fig_line.add_trace(go.Scatter(
            x=transfer_dates_df["date"],
            y=transfer_dates_df["y_val"],
            mode="markers",
            name="Depotübertrag",
            marker=dict(symbol="diamond", size=10, color="orange", line=dict(width=1.2, color="black")),
            text=transfer_dates_df["hover_text"],
            hoverinfo="text"
        ))

    # Legende nach unten schieben
    fig_line.update_layout(
        yaxis_title="EUR", 
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", yanchor="top")
    )
    st.plotly_chart(fig_line, width="stretch")

# ---------------------------------------------------------------------------
# ETF LOOK-THROUGH
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 ETF Look-Through: Einzelaktien-Exposure")
st.caption(
    "Schlüsselt deine Positionen auf die zugrundeliegenden Einzelaktien auf. "
    "Wähle 'Gesamtes Portfolio' für eine konsolidierte Sicht über alle ETFs und Direktaktien hinweg."
)

etf_options = open_df[["ISIN", "Name", "Ticker", "Aktueller Wert"]].dropna(subset=["Aktueller Wert"])
if etf_options.empty:
    st.info("Keine Positionen mit Kursdaten verfügbar.")
else:
    look_through_choices = ["Gesamtes Portfolio (kumuliert)"] + list(
        etf_options.apply(lambda r: f"{r['Name']} ({r['ISIN']})", axis=1)
    )
    selected_choice = st.selectbox("Portfolio / ETF auswählen", options=look_through_choices)

    if selected_choice == "Gesamtes Portfolio (kumuliert)":
        aggregated_holdings = []
        sources_used = {}
        total_portfolio_val = open_df["Aktueller Wert"].sum()

        for _, row in etf_options.iterrows():
            isin = row["ISIN"]
            name = row["Name"]
            ticker = row["Ticker"]
            pos_val = row["Aktueller Wert"]

            holdings_df = pd.DataFrame()
            source = ""
            
            if isin in ISIN_HOLDINGS_FALLBACK or "ETF" in name or "MSCI" in name or "World" in name or "All Country" in name or "FUND" in str(row.get("asset_class", "")):
                holdings_df, source = fetch_etf_holdings(ticker, isin)

            if not holdings_df.empty:
                holdings_df = holdings_df.copy()
                w_max = holdings_df["Weight"].max()
                if w_max > 1.0:
                    holdings_df["Weight"] = holdings_df["Weight"] / 100.0

                sources_used[name] = source
                for _, h_row in holdings_df.iterrows():
                    aggregated_holdings.append({
                        "Name": h_row["Name"],
                        "Symbol": h_row["Symbol"],
                        "Investiert (€)": h_row["Weight"] * pos_val
                    })
            else:
                aggregated_holdings.append({
                    "Name": name,
                    "Symbol": ticker if ticker else isin,
                    "Investiert (€)": pos_val
                })

        if aggregated_holdings:
            agg_df = pd.DataFrame(aggregated_holdings)
            agg_df = agg_df.groupby(["Name", "Symbol"], as_index=False)["Investiert (€)"].sum()
            agg_df["Gewichtung (%)"] = (agg_df["Investiert (€)"] / total_portfolio_val) * 100
            agg_df = agg_df.sort_values("Investiert (€)", ascending=False).reset_index(drop=True)

            source_str = ", ".join([f"{k} ({v})" for k, v in sources_used.items()])
            st.caption(f"Verwendete ETF-Quellen: {source_str if source_str else 'Keine (nur Direktaktien)'}")

            hcol1, hcol2 = st.columns([1.3, 1])
            with hcol1:
                fig_bar = px.bar(agg_df.head(15), x="Investiert (€)", y="Name", orientation="h", text="Gewichtung (%)")
                fig_bar.update_traces(texttemplate="%{text:.2f} %", textposition="outside")
                fig_bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10))
                st.plotly_chart(fig_bar, width="stretch")
            with hcol2:
                st.dataframe(
                    agg_df[["Name", "Symbol", "Gewichtung (%)", "Investiert (€)"]].style.format(
                        {"Gewichtung (%)": "{:.2f}", "Investiert (€)": "{:,.2f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

    else:
        selected_isin = selected_choice.split("(")[-1].rstrip(")")
        row = etf_options[etf_options["ISIN"] == selected_isin].iloc[0]

        holdings_df, source = fetch_etf_holdings(row["Ticker"], selected_isin)

        if holdings_df.empty:
            st.warning(f"Für {row['Name']} ({selected_isin}) konnten keine Holdings-Daten gefunden werden.")
        else:
            st.caption(f"Datenquelle: {source}")
            holdings_df = holdings_df.copy()
            w_max = holdings_df["Weight"].max()
            if w_max > 1.0:
                holdings_df["Weight"] = holdings_df["Weight"] / 100.0

            holdings_df["Investiert (€)"] = holdings_df["Weight"] * row["Aktueller Wert"]
            holdings_df["Gewichtung (%)"] = holdings_df["Weight"] * 100
            holdings_df = holdings_df.sort_values("Investiert (€)", ascending=False)

            hcol1, hcol2 = st.columns([1.3, 1])
            with hcol1:
                fig_bar = px.bar(holdings_df.head(15), x="Investiert (€)", y="Name", orientation="h", text="Gewichtung (%)")
                fig_bar.update_traces(texttemplate="%{text:.2f} %", textposition="outside")
                fig_bar.update_layout(yaxis=dict(autorange="reversed"), margin=dict(t=10))
                st.plotly_chart(fig_bar, width="stretch")
            with hcol2:
                st.dataframe(
                    holdings_df[["Name", "Symbol", "Gewichtung (%)", "Investiert (€)"]].style.format(
                        {"Gewichtung (%)": "{:.2f}", "Investiert (€)": "{:,.2f}"}
                    ),
                    width="stretch",
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# KONSOLIDIERTE DETAIL-DIAGNOSE
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Diagnose-Tool: Transaktions-Details")
st.caption("Analysiere die lückenlose Historie aller Transaktionen und Buchungen pro Wertpapier.")

name_groups = {}
for is_id, raw_name in isin_names.items():
    norm_name = normalize_company_name(raw_name)
    if norm_name not in name_groups:
        name_groups[norm_name] = []
    name_groups[norm_name].append(is_id)

dropdown_options = []
display_to_isins = {}
for norm_name, isins in sorted(name_groups.items()):
    label = f"{norm_name} ({' / '.join(isins)})"
    dropdown_options.append(label)
    display_to_isins[label] = isins

selected_label = st.selectbox("Wertpapier für Detail-Transaktionen auswählen", options=dropdown_options)
if selected_label:
    selected_isins = display_to_isins[selected_label]
    debug_df = tx[tx["ISIN"].isin(selected_isins)].sort_values("date")
    st.dataframe(
        debug_df[["date", "ISIN", "Name", "type", "shares", "price", "amount", "fee", "tx_id"]],
        width="stretch",
        hide_index=True
    )

# ---------------------------------------------------------------------------
# VERGANGENE INVESTMENTS
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📜 Vergangene Investments")
st.caption("Positionen, die du komplett verkauft hast (berechnet nach der Durchschnittskosten-Methode).")

if closed_df.empty:
    st.info("Noch keine vollständig geschlossenen Positionen vorhanden.")
else:
    closed_display = closed_df.copy()
    closed_display["Zeitraum"] = (
        closed_display["first_date"].dt.strftime("%d.%m.%Y") + " – " + closed_display["last_date"].dt.strftime("%d.%m.%Y")
    )
    closed_display = closed_display.rename(
        columns={
            "total_invested": "Investiert gesamt (€)",
            "total_proceeds": "Erlös gesamt (€)",
            "realized_pl": "Realisierter G/V (€)",
            "realized_pl_pct": "Realisierter G/V (%)",
        }
    ).sort_values("Realisierter G/V (€)", ascending=False)

    cols = ["Name", "ISIN", "Zeitraum", "Investiert gesamt (€)", "Erlös gesamt (€)", "Realisierter G/V (€)", "Realisierter G/V (%)"]
    st.dataframe(
        closed_display[cols].style.format(
            {
                "Investiert gesamt (€)": "{:,.2f}",
                "Erlös gesamt (€)": "{:,.2f}",
                "Realisierter G/V (€)": "{:,.2f}",
                "Realisierter G/V (%)": "{:.2f}",
            }
        ).map(color_profit_loss, subset=["Realisierter G/V (%)"]), # Farbfunktion statt background_gradient
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# 📊 HISTORISCHE ZINSESZINS-ANALYSE (FINANZFLUSS-STYLE)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Historische Zinseszins-Analyse")
st.caption("Visualisiert die tatsächliche historische Entwicklung deines Gesamtvermögens aufgeteilt in Einzahlungen und Zinseszins-Gewinne.")

if not value_history.empty:
    # Monatliche Ansicht für maximale Balken-Dichte
    try:
        hist_sampled = value_history.resample("ME").last().dropna()
    except ValueError:
        hist_sampled = value_history.resample("M").last().dropna()
        
    if not hist_sampled.empty:
        df_hist_analysis = pd.DataFrame(index=hist_sampled.index)
        df_hist_analysis["Einzahlungen"] = hist_sampled["Eingezahltes Kapital"]
        df_hist_analysis["Gesamtkapital"] = hist_sampled["Portfolio-Wert"]
        df_hist_analysis["Datum_Label"] = df_hist_analysis.index.strftime("%m / %Y")
            
        # Berechne die monatlichen Netto-Einzahlungen
        df_hist_analysis["Netto_Einzahlung"] = df_hist_analysis["Einzahlungen"].diff().fillna(df_hist_analysis["Einzahlungen"].iloc[0])
        
        # Berechne den IZF/XIRR als Zinssatz (default 7% falls nicht vorhanden)
        r_rate = float(izf_val) if (izf_val is not None and izf_val > 0) else 0.07
        
        simple_interest_list = []
        compound_interest_list = []
        
        for i, date_i in enumerate(df_hist_analysis.index):
            simple_part_sum = 0.0
            total_part_sum = 0.0
            
            for j in range(i + 1):
                deposit_amount = df_hist_analysis["Netto_Einzahlung"].iloc[j]
                deposit_date = df_hist_analysis.index[j]
                years_elapsed = (date_i - deposit_date).days / 365.25
                
                if years_elapsed > 0:
                    # Exponentieller Zinseszins-Faktor: (1+r)^t - 1
                    f_total = (1 + r_rate)**years_elapsed - 1
                    # Linearer Zins-Faktor: r * t
                    f_simple = r_rate * years_elapsed
                    
                    if f_total > 1e-6:
                        simple_part_sum += deposit_amount * f_simple
                        total_part_sum += deposit_amount * f_total
            
            # Tatsächlichen realen Gewinn ermitteln (jetzt nativ perfekt!)
            actual_gain = df_hist_analysis["Gesamtkapital"].iloc[i] - df_hist_analysis["Einzahlungen"].iloc[i]
            
            if actual_gain > 0:
                years_since_start = (date_i - df_hist_analysis.index[0]).days / 365.25
                if years_since_start <= 1.0:
                    simple_val = actual_gain
                    comp_val = 0.0
                elif total_part_sum > 0:
                    # Berechne das exakte mathematische Verhältnis von einfachem zu exponentiellem Wachstum
                    ratio_simple = min(1.0, simple_part_sum / total_part_sum)
                    simple_val = actual_gain * ratio_simple
                    comp_val = actual_gain * (1.0 - ratio_simple)
                else:
                    simple_val = actual_gain
                    comp_val = 0.0
            else:
                simple_val = max(0.0, actual_gain)
                comp_val = 0.0
                
            simple_interest_list.append(simple_val)
            compound_interest_list.append(comp_val)
            
        df_hist_analysis["Einfache_Zinsen"] = simple_interest_list
        df_hist_analysis["Zinseszins"] = compound_interest_list
        
        # Letzten Datenpunkt für die Kacheln auslesen
        latest_row = df_hist_analysis.iloc[-1]
        total_gains_latest = latest_row["Gesamtkapital"] - latest_row["Einzahlungen"]
        
        st.markdown("<br>", unsafe_allow_html=True)
        sum_col1, sum_col2, sum_col3 = st.columns(3)
        sum_col1.metric("Aktuelles Vermögen (Ist-Stand)", f"{latest_row['Gesamtkapital']:,.2f} €")
        sum_col2.metric("Deine Einzahlungen gesamt", f"{latest_row['Einzahlungen']:,.2f} €")
        sum_col3.metric("Erwirtschafteter Ertrag gesamt", f"{total_gains_latest:,.2f} €")
        
        # Drei Stacked-Traces im Plotly-Finanzfluss-Style
        fig_hist_zins = go.Figure()
        
        # Stack 1: Einzahlungen (Finanzfluss-Blau)
        fig_hist_zins.add_trace(go.Bar(
            name="Einzahlungen",
            x=df_hist_analysis["Datum_Label"],
            y=df_hist_analysis["Einzahlungen"],
            marker_color="rgba(59, 130, 246, 0.85)",  
            hovertemplate="%{y:,.2f} €<extra></extra>"
        ))
        
        # Stack 2: Einfache Zinsen (Helleres Orange / linearer Ertrag)
        fig_hist_zins.add_trace(go.Bar(
            name="Einfache Zinsen",
            x=df_hist_analysis["Datum_Label"],
            y=df_hist_analysis["Einfache_Zinsen"],
            marker_color="rgba(253, 186, 116, 0.85)",  
            hovertemplate="%{y:,.2f} €<extra></extra>"
        ))
        
        # Stack 3: Zinseszins (Finanzfluss-Orange / Zins-auf-Zins)
        fig_hist_zins.add_trace(go.Bar(
            name="Zinseszins (Schneeballeffekt)",
            x=df_hist_analysis["Datum_Label"],
            y=df_hist_analysis["Zinseszins"],
            marker_color="rgba(249, 115, 22, 0.85)",  
            hovertemplate="%{y:,.2f} €<extra></extra>"
        ))
        
        fig_hist_zins.update_layout(
            barmode="stack",
            hovermode="x unified",
            xaxis_title="Datum",
            yaxis_title="Wert in EUR",
            xaxis=dict(tickmode="auto", nticks=10, tickangle=-45),  # Automatische, perfekt entzerrte Beschriftung
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center", yanchor="top"),
            margin=dict(t=10, b=10)
        )
        
        st.plotly_chart(fig_hist_zins, width="stretch")
    else:
        st.info("Nicht genügend Datenpunkte für eine historische Analyse vorhanden.")
else:
    st.info("Kurshistorie wird geladen, um die historische Zinseszins-Analyse anzuzeigen.")
