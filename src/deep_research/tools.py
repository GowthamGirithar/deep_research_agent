from utils import get_current_dir
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from pydantic import BaseModel, Field


# MCP server configuration for filesystem access
# we will store the data in the file
# we can have array of config as it is passed to the MultiServerMCPClient
# we can use Tavily search API , to keep it simple using the mcp file system with keeping some data in file.
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",  # Auto-install if needed
            "@modelcontextprotocol/server-filesystem",
            str(get_current_dir() / "files")  # Path to research documents
        ],
        "transport": "stdio"  # Communication via stdin/stdout
    }
}

# Global client variable - will be initialized lazily
_client = None

def get_mcp_client():
    global _client
    if _client is None:
        _client = MultiServerMCPClient(mcp_config)
    return _client


@tool(parse_docstring=True)
# this helps us knowing what LLM decide at each step- we explicity mention it in the prompt to call this after search results
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool
# ConductResearch execution is nothing but research agent workflow execution
class ConductResearch(BaseModel):
    """
    Tool for the supervisor to delegate research tasks to a sub-agent.

    Attributes:
        research_question (str): The specific question or topic to be researched by the sub-agent.
    """
    research_question: str = Field(
        description="The specific question or topic to be researched by the sub-agent."
    )

@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass
    