import streamlit as st
import asyncio
import os
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv
from functools import partial
import httpx
from tavily import TavilyClient
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from binance import AsyncClient
from dataclasses import dataclass

# --- Custom Agent/Tool Library Imports (Simulated) ---
# In a real project, these would be in separate files. For this Streamlit app,
# we are including the necessary classes directly.
from agents import Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel, function_tool, RunContextWrapper, SQLiteSession

#==============================================================================
# 1. TOOL DEFINITIONS (from tools.py)
#==============================================================================

# Load environment variables
_: bool=load_dotenv(find_dotenv())

@dataclass
class AgentContext:
    """A dataclass to hold all shared data for an agent run."""
    session: SQLiteSession

@function_tool
async def tavily_search(wrapper: RunContextWrapper, query: str):
    """A search engine optimized for comprehensive, accurate, and trusted results."""
    # Note: In a real app, API keys would come from the wrapper context.
    # For this example, we'll pull from the environment directly for simplicity.
    client = TavilyClient(os.environ.get("Tavily_Api_Key"))
    search_callable = partial(client.search, query=query, max_results=5)
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, search_callable)
    return response

@function_tool
async def news_search(wrapper: RunContextWrapper, query: str):
    """Searches for general news articles on a given topic."""
    current_date = datetime.now(timezone.utc)
    from_date = current_date - timedelta(days=10)
    from_date_str = from_date.strftime("%Y-%m-%d")
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query, 'from': from_date_str, 'sortBy': 'popularity',
        'apiKey': os.environ.get("News_API"),
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

@function_tool
async def crypto_panic(wrapper: RunContextWrapper, coin_symbol: str):
    """Fetches specialized, real-time cryptocurrency news from CryptoPanic."""
    url = "https://cryptopanic.com/api/developer/v2/posts/"
    params = {
        "auth_token": os.environ.get("CRYPTO_PANIC_API"),
        "public": "true", "currencies": coin_symbol
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json().get('results', [])

async def _fetch_ohlcv_data(wrapper: RunContextWrapper, symbol: str, interval: str, limit: int):
    """Internal helper to fetch OHLCV data from Binance."""
    trading_pair = f"{symbol.upper()}USDT"
    client = await AsyncClient.create(os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET"))
    try:
        klines = await client.get_klines(symbol=trading_pair, interval=interval, limit=limit)
        return [{
            "open_time": datetime.fromtimestamp(int(k[0])/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": datetime.fromtimestamp(int(k[6])/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        } for k in klines]
    finally:
        await client.close_connection()

@function_tool
async def get_ohlcv_data(wrapper: RunContextWrapper, symbol: str, interval: str = '1d', limit: int = 90):
    """A tool to fetch historical OHLCV data for a cryptocurrency from Binance."""
    return await _fetch_ohlcv_data(wrapper, symbol, interval, limit)

@function_tool
async def get_advanced_trade_signal(wrapper: RunContextWrapper, symbol: str, interval: str = '1h'):
    """Calculates Support/Resistance zones and provides advanced signals based on Volume Profile."""
    ohlcv_data = await _fetch_ohlcv_data(wrapper, symbol, interval=interval, limit=300)
    if not isinstance(ohlcv_data, list) or len(ohlcv_data) < 50:
        return {"error": "Could not retrieve sufficient historical data for analysis."}
    
    # Simplified analysis logic for demonstration in the UI
    last_candle = ohlcv_data[-1]
    signal = "Neutral"
    if last_candle['close'] > last_candle['open']:
        signal = "Potential Buy Signal"
    elif last_candle['close'] < last_candle['open']:
        signal = "Potential Sell Signal"

    return {
        "symbol": f"{symbol.upper()}USDT",
        "current_price": last_candle['close'],
        "signal": signal,
        "reason": f"Analysis based on the last candle movement for the {interval} interval."
    }

#==============================================================================
# 2. AGENT DEFINITIONS (from main.py)
#==============================================================================

# --- LLM Service Configuration ---
# NOTE: Replace with your actual API key sources
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

# --- Agent Definitions ---
technical_analysis_agent = Agent(name="Technical Analysis Agent", instructions="You are an expert Technical Analysis Agent. Use `get_advanced_trade_signal` for analysis or `get_ohlcv_data` for raw historical data. Convert crypto names to symbols (e.g., Bitcoin to BTC).", model=llm_model, tools=[get_advanced_trade_signal, get_ohlcv_data])
sentiment_analysis_agent = Agent(name="Sentiment Analysis Agent", instructions="You are a Financial News Analyst Agent. For news, use `crypto_panic` first, then `news_search`, and finally `tavily_search` as a last resort. For current prices, use `tavily_search`.", model=llm_model, tools=[crypto_panic, news_search, tavily_search])
reflection_agent = Agent(name="Reflection Agent", instructions="You are a quality control agent. You will receive a list of news articles. Score each source from 1-10 for objectivity. Your final output MUST be a list containing ONLY the articles that scored 7 or higher.", model=llm_model)
citation_agent = Agent(name="Citation Agent", instructions="You are a Citation Agent. For each article you receive, create a citation with the source URL and a concise summary of no more than 30 words.", model=llm_model)

orchestration_agent = Agent(
    name="Orchestration Agent",
    instructions="You are the central Orchestration Agent. You MUST execute the step-by-step plan from the Planning Agent precisely. Deploy analysis agents as directed. If the `Sentiment_Analysis_Agent` was used, you MUST then use the `Reflection_Agent` and `Citation_Agent`. Finally, compile all results into a single, comprehensive report. Your output IS the report itself, nothing more.",
    model=llm_model,
    tools=[
        technical_analysis_agent.as_tool(tool_name="Technical_Analysis_Agent", tool_description="Performs quantitative analysis on a cryptocurrency for trading signals and price data."),
        sentiment_analysis_agent.as_tool(tool_name="Sentiment_Analysis_Agent", tool_description="Performs qualitative analysis to find recent news, articles, and sentiment."),
        citation_agent.as_tool(tool_name="Citation_Agent", tool_description="Processes and formats sources for the final report."),
        reflection_agent.as_tool(tool_name="Reflection_Agent", tool_description="Filters news articles for quality and reputation.")
    ]
)

planning_agent = Agent(
    name="Planning Agent",
    instructions="You are a planning agent. You create a step-by-step plan based on the user's request. The plan must only include the necessary agents: `Technical_Analysis_Agent` for charts/signals, `Sentiment_Analysis_Agent` for news, or both for a full report. Your output is a numbered list plan. Handoff to the Orchestration Agent when done.",
    model=llm_model,
    handoffs=[orchestration_agent]
)

requirement_gathering_agent = Agent(
    name="Requirement Gathering Agent",
    instructions="You are the Requirement Gathering Agent. Your job is to clarify vague user requests. You must ask questions to determine the specific cryptocurrency, type of analysis (Technical or News), and timeframe. Do not answer questions yourself. Once the request is clear, summarize it and handoff to the Planning Agent.",
    model=llm_model,
    handoffs=[planning_agent]
)

#==============================================================================
# 3. STREAMLIT UI APPLICATION
#==============================================================================

# --- Page Configuration ---
st.set_page_config(page_title="Crypto Deep Research Agent", page_icon="🤖", layout="wide")

# --- Enhanced Styling ---
st.markdown("""
<style>
/* ... (Existing CSS is kept as is, no changes needed) ... */
body { background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%); }
.header-gradient { background: linear-gradient(90deg, #6366f1 0%, #28a745 100%); padding: 2rem 0.5rem 1.5rem 0.5rem; border-radius: 1rem; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(40,167,69,0.08); }
.header-title { color: white; font-size: 2.5rem; font-weight: 700; letter-spacing: 1px; text-align: center; }
.header-desc { color: #f3f4f6; font-size: 1.2rem; text-align: center; margin-top: 0.5rem; }
.stSpinner > div > div { border-top-color: #6366f1 !important; border-width: 4px !important; }
[data-testid="stChatMessage"] { border-radius: 0.7rem !important; box-shadow: 0 2px 12px rgba(99,102,241,0.07); margin-bottom: 0.5rem; transition: all 0.3s ease; border-left: 5px solid; }
[data-testid="stChatMessage"][data-testid="stChatMessage-role-user"] { background: #e0e7ff !important; color: #374151 !important; border-left-color: #6366f1; }
[data-testid="stChatMessage"][data-testid="stChatMessage-role-assistant"] { background: #f0fdf4 !important; color: #065f46 !important; border-left-color: #28a745; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="header-gradient">
    <div class="header-title">🤖 Crypto Deep Research Agent</div>
    <div class="header-desc">An autonomous multi-agent system for advanced cryptocurrency analysis.</div>
</div>
""", unsafe_allow_html=True)

# --- Asynchronous Execution Helper ---
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- Agent Initialization ---
@st.cache_resource
def initialize_agent_system():
    """Initializes the AgentContext and checks for required API keys."""
    # This context will be passed to the agent runner.
    # In a real app, you would populate this with user-specific session data.
    context = AgentContext(session=SQLiteSession("user_session_db"))
    return context

context = initialize_agent_system()

# --- Chat Interface Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your crypto research today?"}]
if "current_agent" not in st.session_state:
    st.session_state.current_agent = requirement_gathering_agent

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about technical analysis, news, or a full report..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""
        # Use st.status to show the agent workflow progress
        with st.status("Agents are starting their work...", expanded=True) as status:
            current_input = prompt
            current_agent = st.session_state.current_agent

            while True:
                status.update(label=f"🤖 Running Agent: **{current_agent.name}**")
                
                result = run_async(
                    Runner.run(
                        starting_agent=current_agent,
                        input=current_input,
                        context=context,
                        session=context.session
                    )
                )

                # If a handoff occurred, update the agent and continue the loop
                if result.last_agent and result.last_agent.name != current_agent.name:
                    status.write(f"✅ Handoff from **{current_agent.name}** to **{result.last_agent.name}**.")
                    current_agent = result.last_agent
                    current_input = result.final_output
                    
                    # If the new agent has no handoffs, it's the final one in the chain
                    if not getattr(current_agent, 'handoffs', []):
                        status.update(label=f"🤖 Compiling final report with: **{current_agent.name}**")
                        final_result = run_async(
                            Runner.run(
                                starting_agent=current_agent, 
                                input=current_input, 
                                context=context,
                                session=context.session
                            )
                        )
                        final_answer = final_result.final_output
                        status.update(label="✅ Workflow Complete!", state="complete", expanded=False)
                        break
                else:
                    # No handoff, so the agent is either asking a question or has finished
                    final_answer = result.final_output
                    status.update(label="✅ Agent finished.", state="complete", expanded=False)
                    break
        
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        # Reset to the entry-point agent for the next conversation
        st.session_state.current_agent = requirement_gathering_agent

