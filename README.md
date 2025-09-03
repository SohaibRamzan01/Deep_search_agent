Crypto-Agent: A Hierarchical Multi-Agent System for Advanced Financial Analysis
1. Abstract
This project presents Crypto-Agent, a sophisticated, autonomous multi-agent system designed for advanced cryptocurrency market analysis. The system addresses the challenge of information overload and complexity in the financial domain by delegating specific research tasks to a hierarchy of specialized AI agents. By programmatically integrating quantitative technical analysis with qualitative sentiment analysis, the system delivers comprehensive, structured, and reliable insights from complex data sources. The modular architecture, built in Python using modern asynchronous libraries, demonstrates a robust application of agentic design principles to solve real-world problems in financial technology.

2. Key Features
Dual-Mode Analysis: Performs both quantitative (OHLCV chart data, volume profile) and qualitative (news, social sentiment) analysis.

Intelligent Task Planning: Dynamically creates a precise execution plan tailored to the user's specific query, preventing unnecessary API calls and agent runs.

Automated End-to-End Workflow: A fully automated pipeline from initial user request clarification to final, synthesized report generation.

Data Source Validation: Implements a quality control mechanism (Reflection Agent) to score and filter news sources for reliability and objectivity, ensuring data integrity.

Modular and Extensible Design: The agent-based architecture allows for easy maintenance and future expansion with new agents or tools.

Natural Language Interface: Users interact with the system through simple, conversational language, making complex analysis accessible.

3. System Architecture
The system is built on a "Chain of Command" architecture, where a user's request is processed through a deterministic pipeline of specialized agents. This modular design ensures a clear separation of concerns, making the system more robust and manageable. Each agent is an independent expert responsible for a single part of the research process.


4. Agent Roles and Responsibilities
Requirement Gathering Agent: Serves as the primary interface to the user. Its sole purpose is to validate and clarify requests to ensure they are actionable before passing them into the system.

Planning Agent: Acts as the system's strategist. It receives the clarified requirements and constructs a targeted, step-by-step plan. It intelligently decides which "worker" agents are needed to fulfill the request efficiently.

Orchestration Agent: The central coordinator. It executes the plan from the Planning Agent, delegates tasks, manages the flow of data between other agents, and synthesizes the final report.

Technical Analysis Agent: A quantitative specialist. It utilizes the Binance API to fetch historical market data (OHLCV), calculates support and resistance zones using a Volume Profile analysis, and generates trade signals.

Sentiment Analysis Agent: A qualitative specialist. It gathers news and articles using a prioritized, three-step fallback system (CryptoPanic -> NewsAPI -> Tavily Search) to ensure comprehensive and relevant data retrieval.

Reflection Agent: A data quality control filter. It programmatically scores news articles based on source reputation and objectivity, discarding low-quality or biased information.

Citation Agent: The final data processor. It receives filtered articles and generates concise summaries (30-word limit) and proper citations for the final report.

6. Setup and Usage
Step 1: Clone the Repository
git clone <your-repository-url>
cd <your-repository-directory>

Step 2: Set Up a Virtual Environment
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Step 3: Install Dependencies
Create a requirements.txt file with all the necessary packages and run:

pip install -r requirements.txt

Step 4: Configure Environment Variables
Create a .env file in the root directory and populate it with your API keys:

# .env file

# LLM API Keys
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

# Tool API Keys
TAVILY_API_KEY="YOUR_TAVILY_API_KEY"

NEWS_API="YOUR_NEWSAPI_KEY"

CRYPTO_PANIC_API="YOUR_CRYPTOPANIC_API_KEY"

# Binance API Keys
BINANCE_API_KEY="YOUR_BINANCE_API_KEY"

BINANCE_API_SECRET="YOUR_BINANCE_SECRET_KEY"

Step 5: Run the Application
Execute the main script from your terminal:

python main.py

The application will then prompt you to enter a query.

7. Example Usage

Example 1: Targeted Technical Analysis

User Input: give me the technical analysis of bitcoin on the 4h interval

System Action: The Planning Agent creates a plan involving only the Technical_Analysis_Agent. The system returns a structured output with price, support/resistance zones, and a trade signal.

Example 2: Comprehensive Report

User Input: Can you give me a full report on Ethereum?


System Action: The Planning Agent creates a multi-step plan involving both technical and sentiment analysis agents, followed by the reflection and citation agents. The system combines all findings into a single, coherent report.
