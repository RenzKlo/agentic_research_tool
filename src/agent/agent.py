"""
Autonomous Agent Implementation

This module implements the core LangChain agent that can autonomously
use tools to accomplish complex tasks.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage
from langchain.tools import BaseTool

from ..tools.file_operations import FileOperationsTool
from ..tools.data_analysis import DataAnalysisTool
from ..tools.web_research import WebResearchTool
from ..tools.code_execution import CodeExecutionTool
from ..tools.api_integration import APIIntegrationTool
from .memory import EnhancedMemory

logger = logging.getLogger(__name__)

class AutonomousAgent:
    """
    LangChain-based autonomous agent that can use tools to accomplish tasks.
    
    The agent can reason about complex problems, break them down into steps,
    select appropriate tools, and execute multi-step workflows autonomously.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tool_calls: int = 20,
        memory_type: str = "buffer",
        tavily_api_key: Optional[str] = None,
        enable_code_execution: bool = False
    ):
        """
        Initialize the autonomous agent.
        
        Args:
            api_key: OpenAI API key
            model: LLM model to use
            temperature: Temperature for generation
            max_tool_calls: Maximum number of tool calls per conversation
            memory_type: Type of memory to use
            tavily_api_key: Tavily API key for web research
            enable_code_execution: Whether to enable code execution
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tool_calls = max_tool_calls
        self.tavily_api_key = tavily_api_key
        self.enable_code_execution = enable_code_execution
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=4000
        )
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Initialize memory
        self.memory = self._initialize_memory(memory_type)
        
        # Create the agent
        self.agent = self._create_agent()
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=max_tool_calls,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            max_execution_time=300  # 5 minutes timeout
        )
        
        logger.info(f"Initialized AutonomousAgent with {len(self.tools)} tools")
    
    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize all available tools for the agent"""
        tools = []
        
        try:
            # File operations tool
            tools.append(FileOperationsTool())
            
            # Data analysis tool
            tools.append(DataAnalysisTool())
            
            # Web research tool (with Tavily API key if available)
            tools.append(WebResearchTool(tavily_api_key=self.tavily_api_key))
            
            # Code execution tool
            tools.append(CodeExecutionTool(enable_execution=self.enable_code_execution))
            
            # API integration tool - temporarily disabled due to Pydantic field issues
            # tools.append(APIIntegrationTool())
            
            logger.info(f"Initialized {len(tools)} tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {str(e)}")
        
        return tools
    
    def _initialize_memory(self, memory_type: str) -> ConversationBufferWindowMemory:
        """Initialize conversation memory"""
        try:
            if memory_type == "enhanced":
                return EnhancedMemory(
                    memory_type="conversation_buffer",
                    max_token_limit=2000,
                    llm=self.llm
                )
            else:
                return ConversationBufferWindowMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    k=10
                )
        except Exception as e:
            logger.error(f"Error initializing memory: {str(e)}")
            return ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=5
            )
    
    def _create_agent(self):
        """Create the LangChain agent with tools"""
        # Define the agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an advanced autonomous AI agent with access to powerful tools. 
            You can accomplish complex tasks by reasoning about problems, breaking them down into steps, 
            and using the appropriate tools available to you.

            Your capabilities include:
            - File operations (read, write, search, analyze files)
            - Data analysis (pandas operations, statistics, visualizations)
            - Web research (search engines, web scraping, information gathering)
            - Code execution (write and run Python code safely)
            - API integration (make API calls, process responses)

            When given a task:
            1. Think step by step about what needs to be done
            2. Break complex tasks into smaller, manageable steps  
            3. Select the most appropriate tools for each step
            4. Execute the steps in logical order
            5. Validate results and adjust approach if needed
            6. Provide clear, comprehensive responses

            Always explain your reasoning and what tools you're using. Be autonomous but transparent.
            If you encounter errors, try alternative approaches. Be thorough but efficient.
            
            Focus on providing value and accomplishing the user's goals completely."""),
            
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
    
    async def arun(self, input_text: str) -> Dict[str, Any]:
        """
        Run the agent asynchronously with the given input.
        
        Args:
            input_text: User input/query
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            logger.info(f"Processing input: {input_text[:100]}...")
            
            # Prepare the input with chat history
            agent_input = {
                "input": input_text,
                "chat_history": self.memory.chat_memory.messages
            }
            
            # Run the agent
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                agent_input
            )
            
            # Extract the response - handle both dict and string formats
            if isinstance(result, dict):
                output = result.get("output", str(result))
                intermediate_steps = result.get("intermediate_steps", [])
            else:
                output = str(result)
                intermediate_steps = []
            
            response = {
                "content": output,
                "tool_calls": self._extract_tool_calls(intermediate_steps),
                "timestamp": datetime.now(),
                "model": self.model
            }
            
            # Save to memory
            self.memory.save_context(
                {"input": input_text},
                {"output": output}
            )
            
            logger.info("Successfully processed input")
            return response
            
        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def run(self, input_text: str) -> Dict[str, Any]:
        """
        Synchronous version of arun.
        
        Args:
            input_text: User input/query
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            logger.info(f"Processing input: {input_text[:100]}...")
            
            # Prepare the input with chat history
            agent_input = {
                "input": input_text,
                "chat_history": self.memory.chat_memory.messages
            }
            
            # Invoke the agent executor
            result = self.agent_executor.invoke(agent_input)
            
            # Handle different result formats
            if isinstance(result, dict):
                # Standard format with output and intermediate_steps
                output = result.get("output", str(result))
                intermediate_steps = result.get("intermediate_steps", [])
            elif isinstance(result, str):
                # Simple string response
                output = result
                intermediate_steps = []
            else:
                # Unknown format
                logger.warning(f"Unexpected result type: {type(result)}")
                output = str(result)
                intermediate_steps = []
            
            response = {
                "content": output,
                "tool_calls": self._extract_tool_calls(intermediate_steps),
                "timestamp": datetime.now(),
                "model": self.model
            }
            
            # Save to memory
            self.memory.save_context(
                {"input": input_text},
                {"output": output}
            )
            
            logger.info("Successfully processed input")
            return response
            
        except Exception as e:
            logger.error(f"Error in agent execution: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    def _extract_tool_calls(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Extract tool call information from intermediate steps"""
        tool_calls = []
        
        for step in intermediate_steps:
            if len(step) >= 2:
                action, observation = step[0], step[1]
                
                tool_call = {
                    "tool": getattr(action, 'tool', 'unknown'),
                    "input": getattr(action, 'tool_input', {}),
                    "output": str(observation)[:500] + "..." if len(str(observation)) > 500 else str(observation),
                    "timestamp": datetime.now()
                }
                tool_calls.append(tool_call)
        
        return tool_calls
    
    def get_conversation_history(self) -> List[BaseMessage]:
        """Get the current conversation history"""
        try:
            return self.memory.chat_memory.messages
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
    
    def clear_memory(self):
        """Clear the conversation memory"""
        try:
            self.memory.clear()
            logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
    
    def add_tool(self, tool: BaseTool):
        """Add a new tool to the agent"""
        try:
            self.tools.append(tool)
            # Recreate agent executor with new tools
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                max_iterations=self.max_tool_calls,
                verbose=True,
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                max_execution_time=300
            )
            logger.info(f"Added tool: {tool.name}")
        except Exception as e:
            logger.error(f"Error adding tool: {str(e)}")
    
    def list_tools(self) -> List[str]:
        """List all available tools"""
        return [tool.name for tool in self.tools]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools"""
        return {tool.name: tool.description for tool in self.tools}
