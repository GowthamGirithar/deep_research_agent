from langchain.chat_models import init_chat_model
from state_agent import AgentState
from prompts import final_report_generation_prompt
from utils import get_today_str
from langchain_core.messages import HumanMessage
import asyncio
from dotenv import load_dotenv


# Load environment variables from .env file

load_dotenv()


writer_model = init_chat_model(model="openai:gpt-4.1", max_tokens=32000)

async def final_report_generation(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """

    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str()
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }