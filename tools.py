import os
from agents import function_tool
from dotenv import load_dotenv, find_dotenv
from functools import partial
import httpx
from tavily import TavilyClient
from agents import RunContextWrapper
from collections import defaultdict
import asyncio
from dataclasses import dataclass
from agents import SQLiteSession
from binance import AsyncClient
from datetime import datetime, timezone, timedelta

_: bool=load_dotenv(find_dotenv())

tavily=os.environ.get("Tavily_Api_Key")
news_api=os.environ.get("News_API")
crypto_panic_api=os.environ.get("CRYPTO_PANIC_API")
binance_api_key=os.environ.get("BINANCE_API_KEY")
binance_api_secret=os.environ.get("BINANCE_API_SECRET")

@dataclass
class AgentContext:
    """A dataclass to hold all shared data for an agent run."""
    session: SQLiteSession
    tavily_api_key: str
    news_api_key: str
    crypto_panic_api_key: str
    binance_api_key: str
    binance_api_secret: str

@function_tool
async def tavily_search(wrapper: RunContextWrapper[AgentContext], query: str):
    """A search engine optimized for comprehensive, accurate, and trusted results."""
    client = TavilyClient(wrapper.context.tavily_api_key)

    search_callable = partial(client.search, query=query)
    loop = asyncio.get_running_loop()
    
    response = await loop.run_in_executor(
        None,
        search_callable
    )
    return response

@function_tool
async def news_search(wrapper: RunContextWrapper[AgentContext], query: str):

    #Dynamic date calculation
    current_date = datetime.now(timezone.utc)
    from_date = current_date - timedelta(days=10)
    from_date_str = from_date.strftime("%Y-%m-%d")

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date_str,
        'sortBy': 'popularity',
        'apiKey': wrapper.context.news_api_key,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            print()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        return {"error": f"API request failed with status {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


@function_tool
async def crypto_panic(wrapper: RunContextWrapper[AgentContext], coin_symbol: str):
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    
    params = {
        "auth_token": wrapper.context.crypto_panic_api_key,
        "public": "true",
        "currencies": coin_symbol
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"Tool is now searching for news on: {coin_symbol}") 
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            news_articles = data.get('results', [])
        
            if not news_articles:
                return f"No news articles found for {coin_symbol}."
            
            return news_articles

    except httpx.HTTPStatusError as e:
        error_message = f"API request failed for {coin_symbol} with status {e.response.status_code}: {e.response.text}"
        return {"error": error_message}
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        return {"error": error_message}


async def _fetch_ohlcv_data(wrapper: RunContextWrapper[AgentContext], symbol: str, interval: str, limit: int):
    """Internal helper function to fetch OHLCV data."""
    trading_pair = f"{symbol.upper()}USDT"
    client = AsyncClient(wrapper.context.binance_api_key, wrapper.context.binance_api_secret)
    try:
        print(f"Fetching OHLCV data for {trading_pair} with interval {interval}...")
        klines = await client.get_klines(symbol=trading_pair, interval=interval, limit=limit)
        
        if not klines:
            return f"No data found for the symbol {trading_pair}."
            
        processed_data = []
        for k in klines:
            processed_data.append({
                "open_time": datetime.fromtimestamp(int(k[0])/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": datetime.fromtimestamp(int(k[6])/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            })
        return processed_data
    finally:
        await client.close_connection()


def calculate_volume_profile_zones(ohlcv_data: list, num_zones: int = 2):
    """
    NEW: Analyzes candle data to find S/R zones based on Volume Profile.
    """
    if not ohlcv_data:
        return {"support_zone": None, "resistance_zone": None}

    volume_bins = defaultdict(float)
    all_prices = [p for candle in ohlcv_data for p in (candle['low'], candle['high'])]
    price_range = max(all_prices) - min(all_prices)
    if price_range == 0: return {"support_zone": None, "resistance_zone": None}
    
    bin_size = price_range / 100

    for candle in ohlcv_data:
        price = candle['low']
        while price <= candle['high']:
            bin_rep = round(price / bin_size) * bin_size
            
            volume_bins[bin_rep] += candle['volume'] / ((candle['high'] - candle['low']) / bin_size + 1)
            price += bin_size

    # Find High-Volume Nodes (HVNs)
    sorted_hvns = sorted(volume_bins.items(), key=lambda item: item[1], reverse=True)
    
    current_price = ohlcv_data[-1]['close']
    
    # Find the strongest HVNs
    support_level = None
    resistance_level = None

    for price, volume in sorted_hvns:
        if price < current_price and support_level is None:
            support_level = price
        if price > current_price and resistance_level is None:
            resistance_level = price
        if support_level and resistance_level:
            break
            
    if not support_level or not resistance_level:
        return {"support_zone": None, "resistance_zone": None}

    # Support and Resistance zones
    support_zone = (support_level - bin_size, support_level + bin_size)
    resistance_zone = (resistance_level - bin_size, resistance_level + bin_size)

    return {"support_zone": support_zone, "resistance_zone": resistance_zone}


def analyze_price_action(ohlcv_data: list, support_zone: tuple, resistance_zone: tuple):
    """
    NEW: Analyzes the last few candles for reversal or breakout patterns with volume.
    """
    last_candle = ohlcv_data[-1]
    prev_candle = ohlcv_data[-2]

    volumes = [c['volume'] for c in ohlcv_data[:-1]] 
    avg_volume = sum(volumes) / len(volumes)
    
    # --- Breakout/Breakdown ---
    # Strong close above resistance with high volume
    if last_candle['close'] > resistance_zone[1] and last_candle['volume'] > avg_volume * 1.5:
        return ("Strong Buy Signal", f"Price broke above resistance at {round(resistance_zone[1], 4)} with high volume.")
    
    # Strong close below support with high volume
    if last_candle['close'] < support_zone[0] and last_candle['volume'] > avg_volume * 1.5:
        return ("Strong Sell Signal", f"Price broke below support at {round(support_zone[0], 4)} with high volume.")

    # --- Reversal/Rejection ---
    # Price touched resistance but closed lower
    if resistance_zone[0] <= last_candle['high'] <= resistance_zone[1] and last_candle['close'] < prev_candle['close']:
        return ("Potential Sell Signal", f"Price rejected from the resistance zone near {round(resistance_zone[0], 4)}.")

    # Price touched support but closed higher
    if support_zone[0] <= last_candle['low'] <= support_zone[1] and last_candle['close'] > prev_candle['close']:
        return ("Potential Buy Signal", f"Price bounced from the support zone near {round(support_zone[1], 4)}.")

    return ("Neutral / Hold", "Price is consolidating between support and resistance zones.")


@function_tool
async def get_ohlcv_data(symbol: str, interval: str = '1d', limit: int = 90):
    """
    A tool to fetch historical OHLCV data for a cryptocurrency.
    """
    try:
        return await _fetch_ohlcv_data(symbol, interval, limit)
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    

@function_tool
async def get_advanced_trade_signal(symbol: str, interval: str = '1h'):
    """
    Calculates S/R zones based on Volume Profile and provides advanced signals.
    """
    try:
        ohlcv_data = await _fetch_ohlcv_data(symbol, interval=interval, limit=300)

        if not isinstance(ohlcv_data, list) or len(ohlcv_data) < 50:
            return {"error": "Could not retrieve sufficient historical data for analysis."}
            
        zones = calculate_volume_profile_zones(ohlcv_data)
        support_zone, resistance_zone = zones.get("support_zone"), zones.get("resistance_zone")
        
        if not support_zone or not resistance_zone:
            return {"signal": "Neutral", "reason": "Could not determine significant support/resistance zones from volume profile."}

        signal, reason = analyze_price_action(ohlcv_data, support_zone, resistance_zone)

        return {
            "symbol": f"{symbol.upper()}USDT",
            "current_price": ohlcv_data[-1]['close'],
            "support_zone (High-Volume)": (round(support_zone[0], 4), round(support_zone[1], 4)),
            "resistance_zone (High-Volume)": (round(resistance_zone[0], 4), round(resistance_zone[1], 4)),
            "signal": signal,
            "reason": reason
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
    
