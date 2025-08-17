from langgraph.graph import StateGraph, START, END
from state_agent import AgentState, AgentInputState
from scope_agent import clarify_with_user, write_research_brief
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from research_supervisor_agent import supervisor_agent
from final_report_generation import final_report_generation
import asyncio

# Load environment variables from .env file
load_dotenv()


checkpointer = InMemorySaver()


deep_reasearch_workflow = StateGraph(AgentState, input_schema= AgentInputState)

# Add workflow nodes
deep_reasearch_workflow.add_node("clarify_with_user", clarify_with_user)
deep_reasearch_workflow.add_node("write_research_brief", write_research_brief)
deep_reasearch_workflow.add_node("supervisor_subgraph", supervisor_agent)
deep_reasearch_workflow.add_node("final_report_generation", final_report_generation)
# Add edges 
deep_reasearch_workflow.add_edge(START, "clarify_with_user")
# deep_reasearch_workflow.add_edge("clarify_with_user", "write_research_brief") - It makes clarify_with_user always go to write_research_brief and does not end even if we provide command goto END
deep_reasearch_workflow.add_edge("write_research_brief", "supervisor_subgraph")
deep_reasearch_workflow.add_edge("supervisor_subgraph", "final_report_generation")
deep_reasearch_workflow.add_edge("final_report_generation", END)

'''
reason for above commented line 

Explicit edges act as “default paths”.

If an edge exists from a node to another, LangGraph ignores your Command(goto=...) for that node — it always follows the edge.

To respect dynamic goto commands (like stopping at END), do not create edges that override them.

'''

# Compile the workflow
deep_research_agent = deep_reasearch_workflow.compile()


async def main():

    thread = {"configurable": {"thread_id": "1"}}
    result = await deep_research_agent.ainvoke({"messages": [HumanMessage(content="I want to research the best coffee shops in San Francisco based on Specialty pour over methods.")]}, config=thread)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
    

