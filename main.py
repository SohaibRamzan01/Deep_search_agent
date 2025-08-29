import os
from agents import Agent, AsyncOpenAI, Runner,OpenAIChatCompletionsModel, handoff
from dotenv import load_dotenv, find_dotenv
import asyncio
from tools import tavily_search, news_search, crypto_panic, rsi_data, macd_data, bollinger_bands_data, get_advanced_trade_signal, get_ohlcv_data


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
technical_analysis_agent: Agent = Agent(
    name="Technical Analysis Agent",
    instructions="You are a helpful assistant that helps people find information related to crypto coin like bicoin, ethereum etc. You have the tool named get_ohlcv_data which you can use to get the historical data for the crypto. You have to ask the user about the coin name, timeframe as intercall and total candlesticks as limit. If the user don't pass all of these 3 parameters then you have to ask him counter question about the timeframe and the total number of candlesticks and then call the tool to provide the response accordingly. You also have the ability to make the signal based on the user query and provide the response to the user. And additionally you also have to just pass the symbol of the crypto coin like if the user ask about the bitcoin then you have to pass the symbol as BTC, for the ethereum you have to pass the symbol as ETH and so on. If the user asks any other query which is not related to any cryptocurrency like bitcoin or ethereum, then simply reply with that i can't help you with that.",
    model=llm_model,
    tools=[get_advanced_trade_signal, get_ohlcv_data])

# Sentiment Analysis Agent
sentiment_analysis_agent: Agent=Agent(
    name="Sentiment Analysis Agent",
    instructions="""You are a helpful assistant. You just have to provide the details related to only crypto related queries. Any other query that the user will ask simply reply with i can't help you with that. If the user ask about the recent details like the current price of xyz crypto or the previous news for the specific crypto coin, then you have to search through the tools that are provided you according to the query like tavily for websearch if the user asked the current price of the crypto coin price, stock price or something like that, and if the user asked about the previous news for the specific crypto coin or the stock then you also have to use other tools (news_search and webz_news) so get the response from these tools and then provide the response to the user and if the user simply ask about other details of which data already available to the llm then you don't have to use the tools for that query, simply provide those details to the user. 
    If the user asks about the news for the specific crypto of stock then your first priority should be to use the tool named news_search, and other 2 tools should be used if this tool don't provide the valid response.
    You also have the ability to calculate and provide the details related to crypto and stock, so provide the response to the user queries related to the investment and other related things.""",
    model=llm_model,
    tools=[crypto_panic, news_search, tavily_search]
    )

# Citation Agent
citation_agent: Agent=Agent(
    name="Citation Agent",
    instructions="""You are a citation agent. Your role is to review the final answer and provide the sources for any information that was gathered from external tools like news searches. You do not have tools of your own; you only process the information given to you.""",
    model=llm_model
)

# ORCHESTRATION AGENT
orchestration_agent: Agent=Agent(
    name="Orchestration Agent",
    instructions="""You are a master orchestrator agent. Your job is to execute a plan provided by the Planning Agent. You will use your specialized sub-agents to perform each step of the plan. First, call the 'Technical Analysis Agent' for price and signal analysis. Second, call the 'Sentiment Analysis Agent' for news and sentiment. Finally, combine the findings and hand them off to the 'Citation Agent' to format the final response.""",
    model=llm_model,
    tools=[
        technical_analysis_agent.as_tool(
            tool_name="Technical_Analysis_Agent", 
            tool_description="Useful for getting technical analysis of crypto coin like bitcoin, ethereum etc."
        ), 
        sentiment_analysis_agent.as_tool(
            tool_name="Sentiment_Analysis_Agent", 
            tool_description="Useful for getting sentiment analysis of crypto coin like bitcoin, ethereum etc."
        ), 
        citation_agent.as_tool(
            tool_name="Citation_Agent", 
            tool_description="Useful for getting citation for the response that you provide to the user."
        )
    ]
)

# PLANNING AGENT
planning_agent: Agent=Agent(
    name="Planning Agent",
    instructions="""You are a planning agent, you have to create a plan according to the requirements that the requirement gathering agent provides you. You have to create a step by step plan, like if the user wants to invest in bitcoin technically and sentimentally for 2 months then you have to create a plan like first you have to get the technical analysis of the bitcoin for 2 months, then you have to get the sentiment analysis of the bitcoin for 2 months, then you have to combine both the analysis and then provide the response back to the user according to his query. You have to create a plan like this according to the requirements that the requirement gathering agent provides you. Once you create the plan then you can handoff the task to the orchestration agent. You should also have an idea what this orchestration agent will do, so you can create the plan accordingly.""",
    model=llm_model,
    handoffs=[orchestration_agent]
)

# REQUIREMENT GATHERING AGENT
requirement_gathering_agent: Agent=Agent(
    name="Requirement Gathering Agent",
    instructions="""You're main focus is to gather clear requirements from the user related to crypto like bitcoin, ethereum, xrp etc. Like if the user inputs a query like i want to invest in crypto then you have to ask counter questions like which crypto coin you want to invest, do you want to get technical analysis or sentiment analysis of that coin, for how long you want to invest and other related questions. You have to ask these questions from the user until you get the clear requirements from the user. This whole agent is actually deep research agent that analyze the crypto coin technically and sentimentally that user inputs and then gives the response back to the user according to his query. You should have an idea what this workflow will do, so you can ask the questions accordingly.
    And also keep in mind that the user can ask any query related to crypto like bitcoin, ethereum etc. So you have to ask the questions accordingly. And if the user's query is not related to crypto then you have to reply with friendly way that this is a deep research agent for crypto only, so the user should ask the questions related to crypto. 
    Once you get the clear requirements from the user then you can handoff the task to the planning agent. And if the user's query is clear and you don't have to ask any counter questions from the user then you can directly handoff the task to the planning agent. You should also have an idea what this planning agent will do, so you can ask the questions accordingly.""",
    model=llm_model,
    handoffs=[planning_agent]
)

# main.py

# ... (all your agent definitions are above this) ...
  
async def run_conversation():
    """
    Runs a conversational loop with the Requirement Gathering Agent until it
    has enough information to hand off to the next agent.
    """
    current_agent = requirement_gathering_agent
    user_input = input("Ask any crypto related question: ")

    while True:
        print(f"\n--- Running Agent: {current_agent.name} ---")
        
        # Run the current agent with the latest input
        result = await Runner.run(starting_agent=current_agent, input=user_input)
        print(result.final_output)
        
        user_input = input("> ")

if __name__ == "__main__":
    asyncio.run(run_conversation())