from langchain.chat_models import init_chat_model
from state_research import ResearchState, ResearcherOutputState
from config import config
from tools import think_tool, get_mcp_client
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from prompts import research_agent_prompt_with_mcp, compress_research_human_message, compress_research_system_prompt
from utils import get_today_str
from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END

# Initialize model using config
model = init_chat_model(model=config.get_research_agent_model(), temperature=0.0)

async def llm_call(state: ResearchState):
    """
    Analyze current state and decide on next actions.

    """

    # Get available tools from MCP server
    client = tools.get_mcp_client()
    mcp_tools = await client.get_tools()

    # Use MCP tools for local document access
    tools = mcp_tools + [think_tool]

    # Initialize model with tool binding
    model_with_tools = model.bind_tools(tools)

    # Process user input with system prompt
    response = model_with_tools.invoke(
        [SystemMessage(content=research_agent_prompt_with_mcp.format(date=get_today_str()))] + 
        state["researcher_messages"] 
    )

    return {
        "researcher_messages": [response]
    }


async def tool_execution_node(state: ResearchState):
     """
     Execute tool calls using MCP tools.

      This node:
    1. Retrieves current tool calls from the last message
    2. Executes all tool calls using async operations (required for MCP)
    3. Returns formatted tool results

    Note: MCP requires async operations due to inter-process communication
    with the MCP server subprocess. This is unavoidable.
     
     """

     tool_calls = state["researcher_messages"][-1].tool_calls

     async def execute_tools():
          # Get fresh tool references from MCP server
        client = get_mcp_client()
        mcp_tools = await client.get_tools()
        tools = mcp_tools + [think_tool]
        tools_by_name = {tool.name: tool for tool in tools}

         # Execute tool calls (sequentially for reliability)
        results = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_calls["name"]]
            if tool_calls["name"] == "think_tool":
                results = tool.invoke(tool_call["args"])
            else:
                results = await tool.ainvoke(tool_call["args"])
            results.append(results)

        # Format results as tool messages
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
            for observation, tool_call in zip(results, tool_calls)
        ]

        return tool_outputs
     
     # Execute tool calls
     tools_results = await execute_tools()

     return {"researcher_messages": tools_results}


def compress_research_finding(state: ResearchState):
     """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for further processing or reporting.

    This function filters out think_tool calls and focuses on substantive
    file-based research content from MCP tools.
    """
     
     researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
     researcher_messages.append(HumanMessage(content=compress_research_human_message))
     
          
     compression_prompt = compress_research_system_prompt.format(date=get_today_str())
     messages = [SystemMessage(content=compression_prompt)] + researcher_messages

     response = model.invoke(messages)

      # Extract raw notes from tool and AI messages
     raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

     return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

def should_continue(state: ResearchState) -> Literal["tool_execution_node", "compress_research_finding"]:
    """Determine whether to continue with tool execution or compress research.

    Determines whether to continue with tool execution or compress research
    based on whether the LLM made tool calls.
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # Continue to tool execution if tools were called
    if last_message.tool_calls:
        return "tool_execution_node"
    # Otherwise, compress research findings
    return "compress_research_finding"



# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
researcher_agent_workflow = StateGraph(ResearchState, output_schema=ResearcherOutputState)

# Add nodes to the graph
researcher_agent_workflow.add_node("llm_call", llm_call)
researcher_agent_workflow.add_node("tool_execution_node", tool_execution_node)
researcher_agent_workflow.add_node("compress_research_finding", compress_research_finding)

# Add edges to connect nodes
researcher_agent_workflow.add_edge(START, "llm_call")
researcher_agent_workflow.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_execution_node": "tool_execution_node",        # Continue to tool execution
        "compress_research_finding": "compress_research_finding",  # Compress research findings
    },
)
researcher_agent_workflow.add_edge("tool_execution_node", "llm_call")  # Loop back for more processing
researcher_agent_workflow.add_edge("compress_research_finding", END)

# Compile the agent
researcher_agent = researcher_agent_workflow.compile()
         











