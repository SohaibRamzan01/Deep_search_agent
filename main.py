import os
from agents import Agent, AsyncOpenAI, Runner,OpenAIChatCompletionsModel, ItemHelpers , SQLiteSession
from dotenv import load_dotenv, find_dotenv
import asyncio
from tools import tavily_search, news_search, crypto_panic, get_advanced_trade_signal, get_ohlcv_data, AgentContext


_: bool=load_dotenv(find_dotenv())
gemini_api_key=os.environ.get("GEMINI_API_KEY")
openai=os.environ.get("Openai_Api_Key")

# LLM Service
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
# LLM Model
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

# Technical Analysis Agent
technical_analysis_agent: Agent[AgentContext] = Agent(
    name="Technical Analysis Agent",
    instructions="""You are an expert **Technical Analysis Agent**. Your sole purpose is to perform detailed, data-driven analysis of cryptocurrency price charts using the specialized tools you have been given.

    **Your Rules of Operation:**
    1.  **Tool Selection:** You have two primary tools. You must choose the correct one based on the user's request:
        - For any request involving an **opinion, analysis, or trading signal** (e.g., "should I buy Bitcoin?", "what's the analysis for ETH?", "give me a trade signal"), you **MUST** use the `get_advanced_trade_signal` tool.
        - If the user asks *only* for **raw historical data or a simple chart** (e.g., "get the daily price for ETH for 90 days"), you should use the `get_ohlcv_data` tool.

    2.  **Parameter Execution:** You are an execution agent. You will receive clear, analyzed tasks. You should **not** ask the user for clarifying information like the coin, interval, or limit, as this is handled by another agent.

    3.  **Symbol Conversion:** You must always convert cryptocurrency names to their official ticker symbols (e.g., Bitcoin to BTC, Ethereum to ETH) before calling any tool.

    4.  **Scope Limitation:** You must only answer questions related to cryptocurrency technical analysis. For any other topic, you must politely state that it is outside your scope of expertise.""",
    model=llm_model,
    tools=[get_advanced_trade_signal, get_ohlcv_data])

# Sentiment Analysis Agent
sentiment_analysis_agent: Agent[AgentContext]=Agent(
    name="Sentiment Analysis Agent",
    instructions="""You are an expert **Financial News and Data Analyst Agent**. Your primary function is to retrieve and present timely news, sentiment, and price data for cryptocurrencies. You must operate with precision and efficiency.

    **Your Rules of Operation:**

    1.  **Scope Limitation:** You must only respond to queries related to specific cryptocurrencies. For any other topic, you must politely state that it is outside your expertise.

    2.  **Internal Knowledge First:** Before using any tool, evaluate if the user's question can be answered from your existing knowledge. Do not use tools for general historical facts or basic definitions (e.g., "What is Bitcoin?").

    3.  **Strict Tool Selection Policy:** When external data is required, you **MUST** follow this precise logic:
        * **For Current Prices:** For any query about the *current price* of an asset, you **MUST** use the `tavily_search` tool.
        * **For News and Recent Events:** For any query about *news, sentiment, or recent events*, you **MUST** follow this exact three-step fallback procedure:
            * **Step 1 (Primary):** Always use the `crypto_panic` tool first. It is your specialized tool for real-time crypto news.
            * **Step 2 (Secondary):** **Only if** `crypto_panic` returns no relevant articles, then use the `news_search` tool for broader, more general news coverage.
            * **Step 3 (Final Fallback):** **Only if both** `crypto_panic` and `news_search` fail to find any results, you must then use `tavily_search` to perform a general web search for the news topic. This is your last resort for news.

    4.  **Synthesize and Respond:** After gathering information, synthesize it into a clear, concise, and helpful answer for the user.""",
    model=llm_model,
    tools=[crypto_panic, news_search, tavily_search]
    )

# Reflection Agent
reflection_agent: Agent[AgentContext] = Agent(
    name="Reflection Agent",
    instructions="""You are a meticulous Reflection Agent acting as a quality control gate. You will receive a list of news articles and you MUST evaluate each one for reputation and objectivity.

    **Your Process:**
    1.  **Score:** Assign a quality score from 1 to 10 to each source. Reputable, objective sources (like Reuters, Bloomberg) get high scores. Biased or promotional sources get low scores.
    2.  **Filter:** Your final output MUST be a list containing ONLY the articles that scored 7 or higher. Do not include the scores themselves.
    
    You have no tools. Your only job is to process, score, and filter.""",
    model=llm_model,
)

# Citation Agent
citation_agent: Agent[AgentContext] =Agent(
    name="Citation Agent",
    instructions="""You are a specialized Citation Agent. You will receive a pre-filtered list of high-quality news articles.

    **Your Task:**
    1.  For each article, create a citation including the source URL.
    2.  Provide a concise summary of each article, **strictly limited to 30 words**.
    3.  Your final output should be a list of these formatted citations and summaries.""",
    model=llm_model
)

# ORCHESTRATION AGENT
orchestration_agent: Agent[AgentContext] =Agent(
    name="Orchestration Agent",
    instructions="""You are the **Orchestration Agent**, the central execution engine and final synthesizer for a multi-agent research system. Your purpose is to execute a formal plan to produce a comprehensive, user-facing report.

    **Your Operational Rules:**

    1.  **Execute the Plan:** Your primary directive is to execute the step-by-step plan you receive from the `Planning Agent`. You **MUST** follow the plan precisely and in the specified order. Do not deviate, skip, or add steps.

    2.  **Deploy Analysis Agents:** The plan will specify which analysis agents to use (`Technical_Analysis_Agent`, `Sentiment_Analysis_Agent`). Execute them as directed to gather the necessary data.

    3.  **Conditional Filtering and Citation:** You MUST check the plan to see if news analysis was performed.
        * **If and ONLY IF** the `Sentiment_Analysis_Agent` was used, you MUST then pass its output to the `Reflection_Agent` for filtering, and subsequently pass the filtered results to the `Citation_Agent`.
        * If the plan **did not** involve the `Sentiment_Analysis_Agent`, you MUST **skip** the reflection and citation steps entirely.

    4.  **Synthesize Findings:** After all the planned analysis tools have been run, you **MUST** compile their outputs into a single, coherent draft of the final answer.

    5.  **Final Output Mandate:** Your final action is to present the compiled report.
        * Your response **MUST** begin *directly* with the analysis (e.g., "Here is the analysis for Bitcoin...").
        * You are **STRICTLY FORBIDDEN** from adding any conversational filler, meta-commentary, or asking follow-up questions (e.g., do not say "I have completed the report"). **Your output IS the report itself, and nothing more.**
    6. If the user asks for technical analysis only than you should not use the Sentiment Analysis Agent and vice versa. Only if the user asks for a comprehensive report you should use both agents.""",
    model=llm_model,
    tools=[
        technical_analysis_agent.as_tool(
            tool_name="Technical_Analysis_Agent", 
            tool_description="Performs quantitative analysis on a cryptocurrency. Use this to get historical price data (OHLCV), volume-based support/resistance zones, and a final trading signal."
        ), 
        sentiment_analysis_agent.as_tool(
            tool_name="Sentiment_Analysis_Agent", 
            tool_description="Performs qualitative analysis. Use this to find recent news, articles, and general sentiment for a cryptocurrency using its specialized search tools."
        ), 
        citation_agent.as_tool(
            tool_name="Citation_Agent", 
            tool_description="Processes and formats sources for the final report. Use this ONLY after filtering articles with the Reflection_Agent."
        ),
        reflection_agent.as_tool(
            tool_name="Reflection_Agent", 
            tool_description="Acts as a quality control filter for news. Use this by passing it the raw output from the 'Sentiment_Analysis_Agent' to receive back only reputable sources."
        )
    ]
)

# PLANNING AGENT
planning_agent: Agent[AgentContext] =Agent(
    name="Planning Agent",
    instructions="""You are a meticulous planning agent. Your sole purpose is to create a step-by-step plan based on the user's specific request.

**Your Core Rules:**
1.  **Analyze the Request:** You MUST carefully analyze the user's query to determine exactly what kind of analysis they want. Do they want **technical analysis**, **sentiment/news analysis**, or **both**?

2.  **Create a Targeted Plan:** Your plan MUST only include the agents necessary to fulfill the user's request.
    - If the user asks for **technical analysis** (e.g., "trade signal for BTC", "chart analysis for ETH"), your plan should ONLY use the `Technical_Analysis_Agent`.
    - If the user asks for **news or sentiment** (e.g., "what's the news on Solana?", "find articles about DOGE"), your plan should ONLY use the `Sentiment_Analysis_Agent` and the subsequent `Reflection_Agent` and `Citation_Agent`.
    - If the user asks for a **comprehensive report** (e.g., "give me a full report on Bitcoin"), then and only then should you create a plan that includes BOTH `Technical_Analysis_Agent` and `Sentiment_Analysis_Agent`.

3.  **Output Format:** Your output MUST be a numbered list outlining the sequence of tasks.
4.  **Handoff:** Once the plan is created, your final action is to `handoff` to the Orchestration Agent.

**Example Plan (for a technical-only request):**
1. Use the `Technical_Analysis_Agent` to get the trade signal for BTC on the 4h interval.
2. Combine the results and present the final analysis.""",
    model=llm_model,
    handoffs=[orchestration_agent]
)

# REQUIREMENT GATHERING AGENT
requirement_gathering_agent: Agent[AgentContext] =Agent(
    name="Requirement Gathering Agent",
    instructions="""You are the Requirement Gathering Agent, the first point of contact for a sophisticated crypto research assistant. Your single most important job is to gather clear, actionable requirements from the user.

Follow these rules strictly:

1. Clarify Vague Requests: If a user's query is not specific (e.g., "I want to invest in crypto"), you must ask clarifying questions. At a minimum, you need to know:
-The specific cryptocurrency (e.g., Bitcoin, Ethereum).
-The type of analysis required (Technical Analysis or Sentiment/News Analysis).
-The relevant timeframe (e.g., for the last 30 days, on the 4-hour chart).

2. Do Not Answer or Analyze: You are strictly forbidden from answering the user's question directly or using any analysis tools. Your only allowed actions are asking questions to the user and handing off the task.

3. Handoff When Ready: Once you have gathered all the necessary details, summarize them clearly and then handoff the task to the Planning Agent. If the user's initial query is already perfectly clear, you can handoff immediately.

4. Stay On Topic: If the user asks about anything other than cryptocurrency, politely state that you can only assist with crypto-related research and decline the request.""",
    model=llm_model,
    handoffs=[planning_agent]
)

async def run_conversation():
    # Create the context object once, containing the session and API keys
    context = AgentContext(
        session=SQLiteSession("user_123"),
        tavily_api_key=os.environ.get("Tavily_Api_Key"),
        news_api_key=os.environ.get("News_API"),
        crypto_panic_api_key=os.environ.get("CRYPTO_PANIC_API"),
        binance_api_key=os.environ.get("BINANCE_API_KEY"),
        binance_api_secret=os.environ.get("BINANCE_API_SECRET")
    )

    current_agent = requirement_gathering_agent
    user_input = input("Ask any crypto related question: ")

    while True:
        print(f"\n--- Running Agent: {current_agent.name} ---")
        
        # Pass the context object into the Runner
        result = await Runner.run(
            starting_agent=current_agent, 
            input=user_input, 
            context=context,
            session=context.session
        )
        
        if result.last_agent and result.last_agent.name != current_agent.name:
            print(f"--- Handoff from {current_agent.name} to {result.last_agent.name} ---")
            
            user_input = result.final_output 
            current_agent = result.last_agent
            
            # Check if the new agent is the final one in the workflow
            if not getattr(current_agent, 'handoffs', []):
                 # FIXED: Use 'context=context' here as well for consistency
                 final_result = await Runner.run(
                     starting_agent=current_agent, 
                     input=user_input, 
                     context=context
                 )
                 print(f"\n--- Workflow Complete ---")
                 print(f"Final Answer: {final_result.final_output}")
                 break
            continue

        else:
            print(f"Agent: {result.final_output}")
            user_input = input("> ")

if __name__ == "__main__":
    asyncio.run(run_conversation())
