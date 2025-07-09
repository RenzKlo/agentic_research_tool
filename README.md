# LangChain Agentic AI - Autonomous Tool-Using Agent

A sophisticated AI agent built with LangChain that can autonomously use tools to accomplish data analysis and research tasks. The agent can reason, plan, and execute multi-step workflows using various tools.

## Features

- 🤖 **Autonomous LLM Agent** - Uses LangChain for tool selection and reasoning
- 🛠️ **Comprehensive Tool Suite** - File operations, data analysis, web search, APIs
- 🧠 **Multi-step Planning** - Agent can break down complex tasks
- 💬 **Interactive Interface** - Modern Streamlit UI for agent interaction
- 📊 **Data Analysis Tools** - Pandas, plotting, statistical analysis
- 🌐 **Web Research** - Search engines, web scraping, API integration
- 💾 **Memory & Context** - Persistent conversation memory
- 🔧 **Extensible Architecture** - Easy to add new tools and capabilities

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run the agent**:
   ```bash
   streamlit run app.py
   ```

## Tools Available to Agent

- **File Operations**: Read, write, search files
- **Data Analysis**: Pandas operations, statistics, visualization
- **Web Research**: Search engines, web scraping
- **API Integration**: REST APIs, external services
- **Code Execution**: Python code runner with safety checks
- **Memory Management**: Conversation history and context

## Architecture

```
├── app.py                 # Main Streamlit interface
├── src/
│   ├── agent/            # Core agent logic
│   │   ├── agent.py      # Main LangChain agent
│   │   ├── tools/        # Tool implementations
│   │   └── memory.py     # Memory management
│   ├── tools/            # Individual tool modules
│   └── utils/            # Utility functions
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Usage Examples

Ask the agent to:
- "Analyze this CSV file and create visualizations"
- "Search for recent papers on machine learning and summarize them"
- "Write a Python script to process this data and save results"
- "Research market trends and create a report"

The agent will autonomously select and use appropriate tools to complete these tasks.

## Development

To add new tools, create a new tool class in `src/tools/` and register it with the agent.

## License

MIT License
