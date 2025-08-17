
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import Optional, Annotated, List, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
import operator

class AgentInputState(MessagesState):
    """MessagesState has the message as field which can be used for the input state"""
    pass


class AgentState(MessagesState):
    """
    Main state for the full deep research 


    Extends MessagesState with additional fields below for the coordination
    MessagesState has the message as field which can be used for the input state
    messages: Annotated[list[AnyMessage], add_messages]
    
    """
    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    # This state field is a list of messages, and when updating, use the add_messages function to merge the new ones in.
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    # Raw unprocessed research notes collected during the research phase
    raw_notes: Annotated[list[str], operator.add] = []
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str


