from state_supervisor import SupervisorState
from langgraph.types import Command
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from prompts import lead_researcher_prompt
from utils import get_today_str
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from tools import think_tool, ConductResearch, ResearchComplete
from langgraph.graph import StateGraph, START, END
from research_agent import researcher_agent
import asyncio
from utils import get_today_str



# Load environment variables from .env file
load_dotenv()

# Initialize model
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)

# System constants
# Maximum number of tool call iterations for individual researcher agents
# This prevents infinite loops and controls research depth per topic
max_researcher_iterations = 6 # Calls to think_tool + ConductResearch

# Maximum number of concurrent research agents the supervisor can launch
# This is passed to the lead_researcher_prompt to limit parallel research tasks
max_concurrent_researchers = 3


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    
    supervisor_messages = state.get("supervisor_messages", [])

    # Prepare system message with current date and constraints
    system_message = lead_researcher_prompt.format(
        date=get_today_str(), 
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )

    messages = [SystemMessage(content=system_message)] + supervisor_messages

    supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
    model_with_tools = model.bind_tools([supervisor_tools])
    response = await model_with_tools.invoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
     """Execute supervisor decisions - either conduct research or end the process.

    Handles:
    - Executing think_tool calls for strategic reflection
    - Launching parallel research agents for different topics
    - Aggregating research results
    - Determining when research is complete

    Args:
        state: Current supervisor state with messages and iteration count

    Returns:
        Command to continue supervision, end process, or handle errors
    """
     supervisor_messages = state.get("supervisor_messages", [])
     research_iterations = state.get("research_iterations", 0)
     most_recent_message = supervisor_messages[-1]

     # Initialize variables for single return pattern
     tool_messages = []
     all_raw_notes = []
     next_step = "supervisor"  # Default next step
     should_end = False

     # check the condition whether it is invoking call more than configured time
     exceeded_iterations=  research_iterations >= max_researcher_iterations
     no_tool_calls = not most_recent_message.tool_calls
     research_complete = any ( tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)

     if exceeded_iterations or no_tool_calls or research_complete:
          should_end = True
          next_step = END

     else:
          # Execute ALL tool calls before deciding next step
        try:
               # Separate think_tool calls from ConductResearch calls
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "think_tool"
            ]

            research_tool_calls= [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ConductResearch"
            ]

            # Handle think_tool calls (synchronous)
            for tool_call in think_tool_calls:
                results = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=results,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # Handle calls (asynchronous)
            if research_tool_calls:
                # Launch parallel research agents
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_question"])
                        ],
                        "research_question": tool_call["args"]["research_question"]
                    }) 
                    for tool_call in research_tool_calls
                ]
            
            # Wait for all research to complete
            tool_results = await asyncio.gather(*coros)
            # Format research results as tool messages
                # Each sub-agent returns compressed research findings in result["compressed_research"]
                # We write this compressed research as the content of a ToolMessage, which allows
                # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
            research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, research_tool_calls)
                ]
            
            tool_messages.extend(research_tool_messages)

            # Aggregate raw notes from all research
            all_raw_notes = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in tool_results
                ]
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

        # Single return point with appropriate state updates
        if should_end:
            return Command(
                goto=next_step,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )
        else:
            return Command(
                goto=next_step,
                update={
                    "supervisor_messages": tool_messages,
                    "raw_notes": all_raw_notes
                }
            )
            
# Build supervisor graph
supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_agent = supervisor_builder.compile()

     

