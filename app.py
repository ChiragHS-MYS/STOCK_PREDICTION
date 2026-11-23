# ---------- DNS RESOLVER PATCH ---------- #
# Fallback to public DNS-over-HTTPS (DoH) or hardcoded Yahoo IPs if local system DNS fails.
import socket
import urllib.request
import json
import threading
import time

real_original_getaddrinfo = socket.getaddrinfo
dns_cache = {}
dns_lock = threading.Lock()
DNS_CACHE_TTL = 300 # Cache DNS results for 5 minutes

def resolve_doh(host):
    try:
        url = f"https://dns.google/resolve?name={host}&type=A"
        req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            data = json.loads(response.read().decode())
            ips = []
            if "Answer" in data:
                for answer in data["Answer"]:
                    if answer.get("type") == 1: # A record
                        ips.append(answer["data"])
            return ips
    except Exception:
        try:
            url = f"https://cloudflare-dns.com/dns-query?name={host}&type=A"
            req = urllib.request.Request(url, headers={"Accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode())
                ips = []
                if "Answer" in data:
                    for answer in data["Answer"]:
                        if answer.get("type") == 1: # A record
                            ips.append(answer["data"])
                return ips
        except Exception:
            return []

def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    cache_key = (host, port, family, type, proto, flags)
    now = time.time()
    
    with dns_lock:
        if cache_key in dns_cache:
            res, expiry = dns_cache[cache_key]
            if now < expiry:
                return res

    try:
        res = real_original_getaddrinfo(host, port, family, type, proto, flags)
        with dns_lock:
            dns_cache[cache_key] = (res, now + DNS_CACHE_TTL)
        return res
    except Exception as orig_exc:
        # If resolution fails for a Yahoo domain, attempt DoH
        if "yahoo" in host.lower():
            ips = resolve_doh(host)
            if ips:
                res = []
                for ip in ips:
                    res.append((socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port)))
                with dns_lock:
                    dns_cache[cache_key] = (res, now + DNS_CACHE_TTL)
                return res
            
            # Hardcoded backup IPs as a last resort fallback
            fallback_ips = []
            if "fc.yahoo.com" in host:
                fallback_ips = ["180.222.114.11", "106.10.236.37", "180.222.114.12", "106.10.236.40"]
            elif "yahoo.com" in host:
                fallback_ips = ["27.123.42.204", "27.123.43.204"]
                
            if fallback_ips:
                res = []
                for ip in fallback_ips:
                    res.append((socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port)))
                with dns_lock:
                    dns_cache[cache_key] = (res, now + DNS_CACHE_TTL)
                return res
                
        raise orig_exc

socket.getaddrinfo = patched_getaddrinfo
# ---------------------------------------- #

import os
import re
import logging
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pymysql
from lstm_model import train_and_predict_lstm, compute_weighted_ensemble
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ---------- FASTAPI APP CONFIG ---------- #
app = FastAPI(title="Stock Prediction App")

print(f"[STARTUP] FastAPI app initializing...")
print(f"[STARTUP] PORT env: {os.getenv('PORT', 'not set')}")
print(f"[STARTUP] MYSQL_URI set: {'yes' if os.getenv('MYSQL_URI') else 'no'}")
print(f"[STARTUP] SECRET_KEY set: {'yes' if os.getenv('SECRET_KEY') else 'no'}")

# MySQL Configuration
mysql_uri = os.getenv("MYSQL_URI")
if not mysql_uri:
    raise ValueError(
        "MYSQL_URI is not set in environment variables. "
        "Please copy .env.example to .env and fill in your MySQL connection string. "
        "See README.md for setup instructions."
    )

secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    raise ValueError(
        "SECRET_KEY is not set in environment variables. "
        "Generate a strong random string and set it as SECRET_KEY. "
        "See README.md for setup instructions."
    )

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Middleware
app.add_middleware(SessionMiddleware, secret_key=secret_key)

# Rate Limiter Setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# MySQL Connection
parsed = urlparse(mysql_uri)
db_host = parsed.hostname
db_port = parsed.port or 3306
db_user = unquote(parsed.username) if parsed.username else parsed.username
db_password = unquote(parsed.password) if parsed.password else parsed.password
db_name = parsed.path.lstrip('/')
if '?' in db_name:
    db_name = db_name.split('?')[0]

def get_db_connection():
    return pymysql.connect(
        host=db_host,
        port=db_port,
        user=db_user,
        password=db_password,
        database=db_name,
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )

db_available = False
try:
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user VARCHAR(255) NOT NULL,
                ticker VARCHAR(50) NOT NULL,
                UNIQUE KEY user_ticker (user, ticker)
            )
        """)
    conn.close()
    print(f"[STARTUP] MySQL database: {db_name}")
    print("MySQL connection and initialization successful!")
    db_available = True
except Exception as e:
    print(f"MySQL connection/initialization failed: {e}")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Templates & Static Files
templates = Jinja2Templates(directory="templates")
if os.path.exists("css"):
    app.mount("/css", StaticFiles(directory="css"), name="css")

# Security Headers & Timing Middleware
@app.middleware("http")
async def add_security_headers_and_timing(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} completed in {elapsed:.2f}s")
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response

# Retry decorator
def retry_api(max_attempts=3, delay=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempts} failed for {func.__name__}: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

# ---------- STOCK DATA ---------- #
# ---------- YAHOO FINANCE CUSTOM SESSION & MAPS ---------- #
yahoo_session = requests.Session()
yahoo_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
})
yahoo_crumb = [None]

def refresh_yahoo_crumb():
    try:
        logger.info("Initializing Yahoo Finance session cookies...")
        r1 = yahoo_session.get("https://fc.yahoo.com", timeout=5)
        logger.info(f"Cookie response code: {r1.status_code}")
        
        r2 = yahoo_session.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=5)
        if r2.status_code == 200:
            yahoo_crumb[0] = r2.text.strip()
            logger.info(f"Yahoo Finance crumb initialized: {yahoo_crumb[0]}")
            return True
        else:
            logger.warning(f"Failed to get Yahoo Finance crumb: {r2.status_code}")
    except Exception as e:
        logger.error(f"Error refreshing Yahoo Finance crumb: {e}")
    return False

YAHOO_SECTOR_MAP = {
    'INFY': 'Technology', 'TCS': 'Technology', 'HCLTECH': 'Technology',
    'WIPRO': 'Technology', 'TECHM': 'Technology', 'LTIM': 'Technology',
    'COFORGE': 'Technology', 'MPHASIS': 'Technology', 'PERSISTENT': 'Technology',
    'HDFCBANK': 'Financial Services', 'ICICIBANK': 'Financial Services',
    'SBIN': 'Financial Services', 'AXISBANK': 'Financial Services',
    'KOTAKBANK': 'Financial Services', 'BAJFINANCE': 'Financial Services',
    'BAJAJFINSV': 'Financial Services', 'HDFCLIFE': 'Financial Services',
    'ICICIGI': 'Financial Services', 'SBILIFE': 'Financial Services',
    'PNB': 'Financial Services', 'BANDHANBNK': 'Financial Services',
    'INDUSINDBK': 'Financial Services', 'YESBANK': 'Financial Services',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy',
    'IOC': 'Energy', 'GAIL': 'Utilities', 'COALINDIA': 'Basic Materials',
    'ADANIGREEN': 'Utilities', 'TATAPOWER': 'Utilities', 'POWERGRID': 'Utilities',
    'NTPC': 'Utilities',
    'HINDUNILVR': 'Consumer Defensive', 'ITC': 'Consumer Defensive',
    'TATACONSUMER': 'Consumer Defensive', 'NESTLEIND': 'Consumer Defensive',
    'BRITANNIA': 'Consumer Defensive', 'DABUR': 'Consumer Defensive',
    'GODREJCP': 'Consumer Defensive', 'MARICO': 'Consumer Defensive',
    'MARUTI': 'Consumer Cyclical', 'TMCV': 'Consumer Cyclical',
    'M&M': 'Consumer Cyclical', 'BAJAJ-AUTO': 'Consumer Cyclical',
    'HEROMOTOCO': 'Consumer Cyclical', 'EICHERMOT': 'Consumer Cyclical',
    'ASHOKLEY': 'Consumer Cyclical', 'TVSMOTOR': 'Consumer Cyclical',
    'SUNPHARMA': 'Healthcare', 'DRREDDY': 'Healthcare', 'CIPLA': 'Healthcare',
    'DIVISLAB': 'Healthcare', 'AUROPHARMA': 'Healthcare', 'LUPIN': 'Healthcare',
    'TORNTPHARM': 'Healthcare', 'BIOCON': 'Healthcare',
    'JSWSTEEL': 'Basic Materials', 'HINDALCO': 'Basic Materials',
    'TATASTEEL': 'Basic Materials', 'VEDL': 'Basic Materials',
    'NATIONALUM': 'Basic Materials', 'SAIL': 'Basic Materials',
    'LT': 'Industrials', 'ULTRACEMCO': 'Basic Materials', 'GRASIM': 'Basic Materials',
    'ADANIPORTS': 'Industrials', 'SHREECEM': 'Basic Materials', 'AMBUJACEM': 'Basic Materials',
    'ACC': 'Basic Materials',
    'BHARTIARTL': 'Communication Services', 'IDEA': 'Communication Services',
    'ZOMATO': 'Consumer Cyclical', 'NYKAA': 'Consumer Cyclical', 'DMART': 'Consumer Defensive',
    'ASIANPAINT': 'Consumer Cyclical', 'PIDILITIND': 'Basic Materials',
    'BERGER': 'Consumer Cyclical', 'DEEPAKNTR': 'Basic Materials',
    'PAGEIND': 'Consumer Cyclical', 'ASTRAZEN': 'Healthcare',
    'DLF': 'Real Estate', 'GODREJPROP': 'Real Estate', 'OBEROIRLTY': 'Real Estate',
    'HAVELLS': 'Industrials', 'SIEMENS': 'Industrials', 'ABB': 'Industrials',
    'BOSCHLTD': 'Consumer Cyclical'
}

# ---------- STOCK DATA ---------- #
_stock_data_cache = {}
_chart_data_cache = {}

def get_stock_data(ticker, cache_bust=None):
    if cache_bust is None:
        cache_bust = datetime.now().strftime('%Y%m%d')
    cache_key = (ticker, cache_bust)
    if cache_key in _stock_data_cache:
        return _stock_data_cache[cache_key]
    
    data = _fetch_stock_data_raw(ticker)
    if data is not None:
        _stock_data_cache[cache_key] = data
    return data

@retry_api(max_attempts=3, delay=2)
def _fetch_stock_data_raw(ticker):
    try:
        if not yahoo_crumb[0]:
            refresh_yahoo_crumb()
            
        for suffix in ['.NS', '.BO']:
            yf_ticker = ticker + suffix
            price = None
            prev_close = None
            long_name = ticker
            high_52 = 'N/A'
            low_52 = 'N/A'
            pe = 'N/A'
            eps = 'N/A'
            market_cap = 'N/A'
            dividend_yield = 'N/A'
            sector = YAHOO_SECTOR_MAP.get(ticker, 'N/A')
            summary = 'No summary available.'
            
            # Step A: Try quoteSummary with crumb
            if yahoo_crumb[0]:
                try:
                    import urllib.parse
                    escaped_symbol = urllib.parse.quote(yf_ticker)
                    url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{escaped_symbol}?modules=financialData,quoteType,defaultKeyStatistics,assetProfile,summaryDetail&crumb={yahoo_crumb[0]}"
                    r = yahoo_session.get(url, timeout=5)
                    if r.status_code == 200:
                        res = r.json().get('quoteSummary', {}).get('result', [])
                        if res:
                            info = res[0]
                            financials = info.get('financialData', {})
                            profile = info.get('assetProfile', {})
                            stats = info.get('defaultKeyStatistics', {})
                            detail = info.get('summaryDetail', {})
                            qtype = info.get('quoteType', {})
                            
                            price = financials.get('currentPrice', {}).get('raw') or detail.get('regularMarketPrice', {}).get('raw')
                            prev_close = detail.get('previousClose', {}).get('raw')
                            long_name = qtype.get('longName', ticker)
                            sector = profile.get('sector', sector)
                            summary = profile.get('longBusinessSummary', summary)[:500]
                            high_52 = detail.get('fiftyTwoWeekHigh', {}).get('raw', 'N/A')
                            low_52 = detail.get('fiftyTwoWeekLow', {}).get('raw', 'N/A')
                            pe = detail.get('trailingPE', {}).get('raw', 'N/A')
                            eps = stats.get('trailingEps', {}).get('raw', 'N/A')
                            raw_mc = detail.get('marketCap', {}).get('raw') or financials.get('marketCap', {}).get('raw')
                            if isinstance(raw_mc, (int, float)) and raw_mc > 0:
                                if raw_mc >= 1e7:
                                    market_cap = f"₹{raw_mc / 1e7:,.2f} Cr"
                                else:
                                    market_cap = f"₹{raw_mc:,.2f}"
                            else:
                                market_cap = 'N/A'
                            
                            raw_div = (
                                detail.get('dividendYield', {}).get('raw')
                                or detail.get('trailingAnnualDividendYield', {}).get('raw')
                                or financials.get('dividendYield', {}).get('raw')
                                or stats.get('trailingAnnualDividendYield', {}).get('raw')
                            )
                            div_rate = detail.get('dividendRate', {}).get('raw') or financials.get('dividendRate', {}).get('raw')
                            
                            if raw_div is not None and isinstance(raw_div, (int, float)) and raw_div > 0:
                                pct = raw_div * 100 if raw_div < 0.5 else raw_div
                                if div_rate and isinstance(div_rate, (int, float)) and div_rate > 0:
                                    dividend_yield = f"{pct:.2f}% (₹{div_rate:.2f}/share)"
                                else:
                                    dividend_yield = f"{pct:.2f}%"
                            elif div_rate and isinstance(div_rate, (int, float)) and div_rate > 0:
                                dividend_yield = f"₹{div_rate:.2f}/share"
                            else:
                                dividend_yield = "0.00% (No dividend)"
                except Exception as e:
                    logger.warning(f"Error fetching quoteSummary for {yf_ticker}: {e}")
                    
            # Step B: Fallback to chart endpoint if price is not found
            if price is None:
                try:
                    import urllib.parse
                    escaped_symbol = urllib.parse.quote(yf_ticker)
                    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{escaped_symbol}?range=1d&interval=1m"
                    r = yahoo_session.get(url, timeout=5)
                    if r.status_code == 200:
                        res = r.json().get('chart', {}).get('result', [])
                        if res:
                            meta = res[0].get('meta', {})
                            price = meta.get('regularMarketPrice')
                            prev_close = meta.get('previousClose')
                            long_name = meta.get('longName', long_name)
                            high_52 = meta.get('fiftyTwoWeekHigh', high_52)
                            low_52 = meta.get('fiftyTwoWeekLow', low_52)
                except Exception as e:
                    logger.warning(f"Error fetching chart fallback for {yf_ticker}: {e}")
                    
            if price is not None:
                if prev_close is not None and prev_close != 0:
                    change = ((price - prev_close) / prev_close) * 100
                else:
                    change = 0.0
                    
                exchange = 'NSE' if suffix == '.NS' else 'BSE'
                return {
                    'ticker': ticker,
                    'symbol': ticker,
                    'name': long_name,
                    'price': round(price, 2),
                    'currentPrice': round(price, 2),
                    'change': round(change, 2),
                    'sector': sector,
                    'summary': summary,
                    'high52': high_52,
                    'low52': low_52,
                    'pe': pe,
                    'eps': eps,
                    'marketCap': market_cap,
                    'dividendYield': dividend_yield,
                    'exchange': exchange
                }
        return None
    except Exception as e:
        logger.error(f"Error getting stock data for {ticker}: {e}")
        return None

# ---------- CHART DATA ---------- #
def get_chart_data(ticker, future_days=30, cache_bust=None):
    if cache_bust is None:
        cache_bust = datetime.now().strftime('%Y%m%d')
    cache_key = (ticker, future_days, cache_bust)
    if cache_key in _chart_data_cache:
        return _chart_data_cache[cache_key]
    
    data = _fetch_chart_data_raw(ticker, future_days=future_days)
    if data:
        _chart_data_cache[cache_key] = data
    return data

@retry_api(max_attempts=3, delay=2)
def _fetch_chart_data_raw(ticker, future_days=30):
    try:
        # Map calendar future_days to trading days & appropriate historical window
        if future_days <= 7:
            trading_days_needed = 5
            history_range = "6mo"
        elif future_days <= 30:
            trading_days_needed = 22
            history_range = "6mo"
        elif future_days <= 90:
            trading_days_needed = 65
            history_range = "1y"
        else: # 365 days
            trading_days_needed = 252
            history_range = "2y"

        for suffix in ['.NS', '.BO']:
            yf_ticker = ticker + suffix
            import urllib.parse
            escaped_symbol = urllib.parse.quote(yf_ticker)
            url = f"https://query2.finance.yahoo.com/v8/finance/chart/{escaped_symbol}?range={history_range}&interval=1d"
            r = yahoo_session.get(url, timeout=5)
            if r.status_code == 200:
                res = r.json().get('chart', {}).get('result', [])
                if res:
                    timestamps = res[0].get('timestamp', [])
                    indicators = res[0].get('indicators', {}).get('quote', [{}])[0]
                    closes = indicators.get('close', [])
                    opens = indicators.get('open', [])
                    highs = indicators.get('high', [])
                    lows = indicators.get('low', [])
                    
                    if not timestamps or not closes:
                        continue
                        
                    df = pd.DataFrame({
                        'Date': pd.to_datetime(timestamps, unit='s'),
                        'Open': opens,
                        'High': highs,
                        'Low': lows,
                        'Close': closes
                    })
                    
                    df.dropna(subset=['Close'], inplace=True)
                    if df.empty:
                        continue
                        
                    df['MA20'] = df['Close'].rolling(window=20).mean()
                    df['Upper'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
                    df['Lower'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
                    df.reset_index(inplace=True)
                    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                    
                    df['Day'] = np.arange(len(df))
                    X = df[['Day']].values
                    y = df['Close'].values
                    model = LinearRegression().fit(X, y)
                    mse = mean_squared_error(y, model.predict(X))
                    std = np.sqrt(mse)
                    
                    last_date = pd.to_datetime(df['Date'].iloc[-1])
                    future_dates = []
                    current_date = last_date
                    
                    while len(future_dates) < trading_days_needed:
                        current_date += timedelta(days=1)
                        if current_date.weekday() < 5:
                            future_dates.append(current_date.strftime('%Y-%m-%d'))
                            
                    future_day_numbers = np.arange(len(df), len(df) + len(future_dates)).reshape(-1, 1)
                    linear_preds = model.predict(future_day_numbers).flatten()
                    
                    # Ensemble with PyTorch LSTM
                    try:
                        lstm_res = train_and_predict_lstm(y, future_days=len(future_dates), epochs=20)
                        ens = compute_weighted_ensemble(
                            linear_preds=linear_preds.tolist(),
                            lstm_preds=lstm_res['lstm_predictions'],
                            linear_std=float(std),
                            lstm_rmse=float(lstm_res['rmse'])
                        )
                        pred_prices = ens['ensemble_predictions']
                        pred_upper = ens['ensemble_upper']
                        pred_lower = ens['ensemble_lower']
                    except Exception as e:
                        logger.warning(f"LSTM/Ensemble prediction error: {e}, falling back to Linear")
                        pred_prices = np.round(linear_preds, 2).tolist()
                        pred_upper = np.round(linear_preds + std, 2).tolist()
                        pred_lower = np.round(linear_preds - std, 2).tolist()
                    
                    data = []
                    for _, row in df.iterrows():
                        data.append({
                            'Date': row['Date'],
                            'Open': round(row['Open'], 2) if 'Open' in row and not pd.isna(row['Open']) else round(row['Close'], 2),
                            'High': round(row['High'], 2) if 'High' in row and not pd.isna(row['High']) else round(row['Close'], 2),
                            'Low': round(row['Low'], 2) if 'Low' in row and not pd.isna(row['Low']) else round(row['Close'], 2),
                            'Close': round(row['Close'], 2),
                            'MA20': round(row['MA20'], 2) if not pd.isna(row['MA20']) else None,
                            'Upper': round(row['Upper'], 2) if not pd.isna(row['Upper']) else None,
                            'Lower': round(row['Lower'], 2) if not pd.isna(row['Lower']) else None,
                            'Predicted': None,
                            'PredictedUpper': None,
                            'PredictedLower': None
                        })
                    for i in range(len(future_dates)):
                        data.append({
                            'Date': future_dates[i],
                            'Open': None,
                            'High': None,
                            'Low': None,
                            'Close': None,
                            'MA20': None,
                            'Upper': None,
                            'Lower': None,
                            'Predicted': round(pred_prices[i], 2),
                            'PredictedUpper': round(pred_upper[i], 2),
                            'PredictedLower': round(pred_lower[i], 2)
                        })
                    return data
        return []
    except Exception as e:
        logger.error(f"Error getting chart data: {e}")
        return []

# ---------- FIXED NEWS DATA (Yahoo Finance Scraper) ---------- #
@lru_cache(maxsize=50)
@retry_api(max_attempts=3, delay=2)
def get_news_data(ticker, cache_bust=datetime.now().strftime('%Y%m%d%H')):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        news_items = soup.select('section[data-test-locator="mega"] h3 a')[:5]
        articles = []
        for item in news_items:
            title = item.get_text(strip=True)
            link = item['href']
            if not link.startswith('http'):
                link = "https://finance.yahoo.com" + link
            articles.append({
                'title': title,
                'source': 'Yahoo Finance',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'description': '',
                'url': link
            })
        return articles
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance news for {ticker}: {e}")
        return []

# ---------- INDICES DATA ---------- #
def get_indices_data():
    """Fetch indices data with fallback to mock data"""
    mock_data = [
        {'name': 'NIFTY 50', 'value': '24,500.00', 'change': 0.5},
        {'name': 'SENSEX', 'value': '80,500.00', 'change': 0.3},
        {'name': 'NIFTY BANK', 'value': '52,000.00', 'change': -0.2},
        {'name': 'INDIA VIX', 'value': '13.50', 'change': 1.2}
    ]
    
    try:
        index_map = {
            'NIFTY 50': '^NSEI',
            'SENSEX': '^BSESN',
            'NIFTY BANK': '^NSEBANK',
            'INDIA VIX': '^INDIAVIX'
        }
        indices = []
        success_count = 0
        
        for name, symbol in index_map.items():
            try:
                import urllib.parse
                escaped_symbol = urllib.parse.quote(symbol)
                url = f"https://query2.finance.yahoo.com/v8/finance/chart/{escaped_symbol}?range=2d&interval=1d"
                r = yahoo_session.get(url, timeout=5)
                if r.status_code == 200:
                    res = r.json().get('chart', {}).get('result', [])
                    if res:
                        meta = res[0].get('meta', {})
                        price = meta.get('regularMarketPrice')
                        prev_close = meta.get('previousClose')
                        
                        timestamps = res[0].get('timestamp', [])
                        closes = res[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
                        valid_data = [(t, c) for t, c in zip(timestamps, closes) if c is not None]
                        
                        if price is None and valid_data:
                            price = valid_data[-1][1]
                        if prev_close is None and len(valid_data) > 1:
                            prev_close = valid_data[-2][1]
                        elif prev_close is None and price is not None:
                            prev_close = price
                            
                        if price is not None:
                            change = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
                            indices.append({'name': name, 'value': f"{price:,.2f}", 'change': round(change, 2)})
                            success_count += 1
                            logger.info(f"✓ Fetched {name}: {price:.2f}")
                            continue
                            
                logger.warning(f"✗ No data for {name}, using fallback")
                fallback = next((item for item in mock_data if item['name'] == name), None)
                if fallback:
                    indices.append(fallback)
            except Exception as idx_error:
                logger.warning(f"✗ Error fetching {name}: {str(idx_error)[:50]}")
                fallback = next((item for item in mock_data if item['name'] == name), None)
                if fallback:
                    indices.append(fallback)
        
        if len(indices) < 4:
            for mock_item in mock_data:
                if not any(idx['name'] == mock_item['name'] for idx in indices):
                    indices.append(mock_item)
        
        logger.info(f"Indices loaded: {success_count} live, {len(indices)-success_count} fallback")
        return indices
            
    except Exception as e:
        logger.error(f"Critical error fetching indices: {str(e)[:100]}")
        return mock_data

# ---------- TRENDING STOCKS ---------- #
def get_trending_stocks():
    sample = [
        'INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM',
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK',
        'RELIANCE', 'ONGC', 'TATASTEEL', 'HINDALCO',
        'MARUTI', 'TMCV', 'M&M',
        'SUNPHARMA', 'DRREDDY', 'CIPLA',
        'HINDUNILVR', 'ITC', 'NESTLEIND',
        'LT', 'BHARTIARTL', 'ASIANPAINT'
    ]
    trending = []
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(get_stock_data, ticker): ticker for ticker in sample}
            for future in as_completed(future_to_ticker, timeout=30):
                ticker = future_to_ticker[future]
                try:
                    stock_data = future.result()
                    if stock_data:
                        trending.append({
                            'ticker': ticker,
                            'name': stock_data['name'],
                            'price': stock_data['price'],
                            'change': stock_data['change']
                        })
                except Exception as e:
                    logger.warning(f"Error fetching trending stock {ticker} in thread pool: {e}")
    except Exception as e:
        logger.error(f"Error in trending stocks ThreadPoolExecutor: {e}")
        
    return trending

# ---------- WATCHLIST HELPERS ---------- #
def get_watchlist_data(tickers):
    return [stock for ticker in tickers if (stock := get_stock_data(ticker))]

# ---------- FASTAPI ROUTES ---------- #

@app.get("/", name="home")
async def home(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    trending = get_trending_stocks()
    return templates.TemplateResponse(
        request=request, name="index.html", context={"trending": trending, "year": datetime.now().year}
    )

@app.get("/all_stocks", name="all_stocks")
async def all_stocks(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="all_stocks.html")

@app.get("/watchlist/test", name="watchlist_test")
async def watchlist_test(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="watchlist_new.html")

@app.get("/watchlist", name="watchlist")
async def watchlist(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    user = request.session["user"]
    user_watchlist = {'tickers': []}
    if db_available:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT ticker FROM watchlists WHERE user = %s", (user,))
                rows = cursor.fetchall()
                user_watchlist['tickers'] = [row['ticker'] for row in rows]
            conn.close()
        except Exception as e:
            logger.error(f"DB error in watchlist: {e}")
    watchlist_data = get_watchlist_data(user_watchlist.get('tickers', []))
    return templates.TemplateResponse(
        request=request, name="watchlist.html", context={"watchlist": watchlist_data}
    )

@app.get("/sector_correlation", name="sector_correlation")
async def sector_correlation(request: Request):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="Sector Correlation.html")

@app.get("/login", name="login")
@limiter.limit("10 per minute")
async def login_get(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@app.post("/login")
@limiter.limit("10 per minute")
async def login_post(request: Request):
    form_data = await request.form()
    username = form_data.get('username')
    password = form_data.get('password')
    if not username or not password:
        return templates.TemplateResponse(request=request, name="login.html", context={"error": "Username and password are required."})
    try:
        user_record = None
        if db_available:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user_record = cursor.fetchone()
            conn.close()
    except Exception:
        logger.exception("Unable to access MySQL during login")
        return templates.TemplateResponse(
            request=request, name="login.html", context={"error": "The database is unavailable. Please try again later."}, status_code=503
        )
    if user_record and check_password_hash(user_record['password'], password):
        request.session['user'] = username
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request=request, name="login.html", context={"error": "Invalid username or password."})

@app.get("/signup", name="signup")
@limiter.limit("5 per minute")
async def signup_get(request: Request):
    return templates.TemplateResponse(request=request, name="signup.html")

@app.post("/signup")
@limiter.limit("5 per minute")
async def signup_post(request: Request):
    form_data = await request.form()
    username = form_data.get('username')
    password = form_data.get('password')
    if not username or not password:
        return templates.TemplateResponse(request=request, name="signup.html", context={"error": "Username and password are required."})
    if not re.match(r'^[a-zA-Z0-9_]{3,20}$', username):
        return templates.TemplateResponse(request=request, name="signup.html", context={"error": "Username must be 3-20 characters, alphanumeric or underscore."})
    try:
        existing_user = None
        if db_available:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                existing_user = cursor.fetchone()
            conn.close()
    except Exception:
        logger.exception("Unable to access MySQL during signup")
        return templates.TemplateResponse(
            request=request, name="signup.html", context={"error": "The database is unavailable. Please try again later."}, status_code=503
        )
    if existing_user:
        return templates.TemplateResponse(request=request, name="signup.html", context={"error": "Username already exists!"})
    hashed_pw = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    try:
        if not db_available:
            raise Exception("Database client not connected")
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
        conn.close()
    except Exception:
        logger.exception("Unable to create user in MySQL")
        return templates.TemplateResponse(
            request=request, name="signup.html", context={"error": "The database is unavailable. Please try again later."}, status_code=503
        )
    request.session['user'] = username
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/logout", name="logout")
async def logout(request: Request):
    request.session.pop('user', None)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@app.get("/analyze", name="analyze")
async def analyze(request: Request, ticker: str = ""):
    if "user" not in request.session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    ticker = ticker.upper()
    if not ticker or not re.match(r'^[A-Z0-9]+$', ticker):
        return templates.TemplateResponse(request=request, name="analysis.html", context={"stock": None, "chart": [], "news": []})
    data = get_stock_data(ticker)
    chart_data = get_chart_data(ticker)
    news_data = get_news_data(ticker)
    return templates.TemplateResponse(request=request, name="analysis.html", context={"stock": data, "chart": chart_data, "news": news_data})

@app.get("/api/trending")
@limiter.limit("100 per hour")
async def api_trending(request: Request):
    if "user" not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    return JSONResponse(get_trending_stocks())

@app.get("/api/indices")
@limiter.limit("100 per hour")
async def api_indices(request: Request):
    if "user" not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    try:
        indices = get_indices_data()
        if not indices or len(indices) == 0:
            mock_data = [
                {'name': 'NIFTY 50', 'value': '24,500.00', 'change': 0.5},
                {'name': 'SENSEX', 'value': '80,500.00', 'change': 0.3},
                {'name': 'NIFTY BANK', 'value': '52,000.00', 'change': -0.2},
                {'name': 'INDIA VIX', 'value': '13.50', 'change': 1.2}
            ]
            return JSONResponse({'indices': mock_data})
        return JSONResponse({'indices': indices})
    except Exception as e:
        logger.error(f"API indices error: {e}")
        mock_data = [
            {'name': 'NIFTY 50', 'value': '24,500.00', 'change': 0.5},
            {'name': 'SENSEX', 'value': '80,500.00', 'change': 0.3},
            {'name': 'NIFTY BANK', 'value': '52,000.00', 'change': -0.2},
            {'name': 'INDIA VIX', 'value': '13.50', 'change': 1.2}
        ]
        return JSONResponse({'indices': mock_data})

@app.get("/api/watchlist")
async def get_watchlist(request: Request):
    if 'user' not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    user = request.session['user']
    watchlist = {'tickers': []}
    if db_available:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT ticker FROM watchlists WHERE user = %s", (user,))
                rows = cursor.fetchall()
                watchlist['tickers'] = [row['ticker'] for row in rows]
            conn.close()
        except Exception as e:
            logger.error(f"DB error in get_watchlist_count: {e}")
    return JSONResponse(get_watchlist_data(watchlist.get('tickers', [])))

@app.get("/api/watchlist/count")
async def get_watchlist_count(request: Request):
    if 'user' not in request.session:
        return JSONResponse({'count': 0})
    user = request.session['user']
    watchlist = {'tickers': []}
    if db_available:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT ticker FROM watchlists WHERE user = %s", (user,))
                rows = cursor.fetchall()
                watchlist['tickers'] = [row['ticker'] for row in rows]
            conn.close()
        except Exception as e:
            logger.error(f"DB error in get_watchlist: {e}")
    return JSONResponse({'count': len(watchlist.get('tickers', []))})

@app.post("/api/watchlist/add")
@limiter.limit("20 per minute")
async def add_to_watchlist(request: Request):
    if 'user' not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    user = request.session['user']
    try:
        body = await request.json()
    except Exception:
        body = {}
    ticker = body.get('ticker', '').upper()
    if not ticker or not re.match(r'^[A-Z0-9]+$', ticker):
        return JSONResponse({'error': 'Invalid ticker format. Use symbols like INFY, TCS.'}, status_code=400)
    stock_data = get_stock_data(ticker)
    if not stock_data:
        return JSONResponse({'error': 'Invalid or unavailable ticker.'}, status_code=400)
    if db_available:
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("INSERT IGNORE INTO watchlists (user, ticker) VALUES (%s, %s)", (user, ticker))
            conn.close()
        except Exception as e:
            logger.error(f"DB error in add_to_watchlist: {e}")
    return JSONResponse({'status': 'added', 'ticker': ticker})

@app.delete("/api/watchlist/{ticker}")
@limiter.limit("20 per minute")
async def remove_from_watchlist(request: Request, ticker: str):
    try:
        if 'user' not in request.session:
            return JSONResponse({'error': 'User not logged in'}, status_code=401)
        
        if not ticker or ticker.strip() == '':
            return JSONResponse({'error': 'Ticker is required'}, status_code=400)
        
        user = request.session.get('user')
        ticker_upper = ticker.upper().strip()
        
        print(f'Removing {ticker_upper} from watchlist for {user}')
        
        if db_available:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                affected_rows = cursor.execute("DELETE FROM watchlists WHERE user = %s AND ticker = %s", (user, ticker_upper))
            conn.close()
            print(f'Update result: modified={affected_rows}')
            if affected_rows == 0:
                return JSONResponse({'error': f'Stock {ticker} not found in watchlist'}, status_code=404)
        
        return JSONResponse({'success': True, 'message': f'{ticker} removed from watchlist'}, status_code=200)
    
    except Exception as e:
        print(f'Error removing from watchlist: {str(e)}')
        return JSONResponse({'error': f'Server error: {str(e)}'}, status_code=500)

@app.get("/api/stocks")
@limiter.limit("50 per hour")
async def api_all_stocks(request: Request):
    if 'user' not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    symbols = [
        'INFY', 'TCS', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'COFORGE', 'MPHASIS', 'PERSISTENT',
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE',
        'ICICIGI', 'SBILIFE', 'PNB', 'BANDHANBNK', 'INDUSINDBK', 'YESBANK',
        'RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL', 'COALINDIA', 'ADANIGREEN', 'TATAPOWER', 'POWERGRID', 'NTPC',
        'HINDUNILVR', 'ITC', 'TATACONSUMER', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'GODREJCP', 'MARICO',
        'MARUTI', 'TMCV', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT', 'ASHOKLEY', 'TVSMOTOR',
        'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'AUROPHARMA', 'LUPIN', 'TORNTPHARM', 'BIOCON',
        'JSWSTEEL', 'HINDALCO', 'TATASTEEL', 'VEDL', 'NATIONALUM', 'SAIL',
        'LT', 'ULTRACEMCO', 'GRASIM', 'ADANIPORTS', 'SHREECEM', 'AMBUJACEM', 'ACC',
        'BHARTIARTL', 'IDEA',
        'ZOMATO', 'NYKAA', 'DMART',
        'ASIANPAINT', 'PIDILITIND', 'BERGER', 'DEEPAKNTR',
        'PAGEIND', 'ASTRAZEN',
        'DLF', 'GODREJPROP', 'OBEROIRLTY',
        'HAVELLS', 'SIEMENS', 'ABB', 'BOSCHLTD'
    ]
    
    stocks = []
    logger.info(f"Starting to fetch {len(symbols)} stocks...")
    
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(get_stock_data, symbol): symbol for symbol in symbols}
            
            for future in as_completed(future_to_symbol, timeout=60):
                symbol = future_to_symbol[future]
                try:
                    stock_data = future.result(timeout=10)
                    if stock_data:
                        stocks.append({
                            'symbol': symbol,
                            'name': stock_data['name'],
                            'sector': stock_data['sector'],
                            'currentPrice': stock_data['price'],
                            'change': stock_data.get('change', 0.0),
                            'peRatio': stock_data['pe']
                        })
                        logger.info(f"✓ Fetched {symbol} - {stock_data['sector']}")
                    else:
                        logger.warning(f"✗ No data for {symbol}")
                except Exception as e:
                    logger.warning(f"✗ Failed to fetch {symbol}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Thread pool error: {e}")
    
    logger.info(f"Successfully fetched {len(stocks)} out of {len(symbols)} stocks")
    return JSONResponse(stocks)

@app.get("/api/news/{ticker}")
@limiter.limit("100 per hour")
async def api_news(request: Request, ticker: str):
    if 'user' not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    ticker = ticker.upper()
    if not ticker or not re.match(r'^[A-Z0-9]+$', ticker):
        return JSONResponse({'error': 'Invalid ticker format. Use symbols like INFY, TCS.'}, status_code=400)
    articles = get_news_data(ticker)
    return JSONResponse(articles)

@app.get("/api/chart/{ticker}")
@limiter.limit("100 per hour")
async def api_chart_data(request: Request, ticker: str, days: int = 30):
    if 'user' not in request.session:
        return JSONResponse({'error': 'Unauthorized'}, status_code=401)
    ticker = ticker.upper()
    if not ticker or not re.match(r'^[A-Z0-9]+$', ticker):
        return JSONResponse({'error': 'Invalid ticker format'}, status_code=400)
    if days not in [7, 30, 90, 365]:
        days = 30
    chart_data = get_chart_data(ticker, future_days=days)
    return JSONResponse(chart_data)

# Exception Handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    logger.warning(f"404 - Not Found: {request.url.path}")
    return templates.TemplateResponse(request=request, name="404.html", status_code=404)

@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    logger.error(f"500 - Server Error: {str(exc)} at {request.url.path}")
    return templates.TemplateResponse(request=request, name="500.html", status_code=500)

if __name__ == '__main__':
    import uvicorn
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5003))
    print(f"\nServer running! Open your browser at: http://localhost:{port} or http://127.0.0.1:{port}\n")
    uvicorn.run("app:app", host=host, port=port, reload=True)
