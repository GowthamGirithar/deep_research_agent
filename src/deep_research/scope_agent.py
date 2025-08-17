from state_agent import AgentState
from langgraph.types import Command
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from state_scope import ClarifyWithUser , ResearchQuestion
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from prompts import clarify_with_user_instructions , transform_messages_into_research_topic_prompt
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from utils import get_today_str

# Load environment variables from .env file
load_dotenv()

# Initialize model
model = init_chat_model(model="openai:gpt-4.1", temperature=0.0)



def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief" , "__end__"]]:
    """
    clarify_with_user is the clarification decision node

    Invoke the LLM with clear prompt to analyze whether we need to ask any more questions to users to have full 
    context to proceed with the research. 

    LLM response is structure with our model and we use that to decide next step

    if we have all the information from user , we proceed to the next node which is write_research_brief
    and if we do not have enough information, we go to end. 

    All these conversation are stored in the agent state using the chekpointer inmemory
    
    """
     # Set up structured output model which is the inbuilt method to return the response in this format
     # prompt has clarified with few shots examples
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # Invoke the model with clarification instructions
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state["messages"]), 
            date=get_today_str()
        ))
    ])
    print(f'The response requires clarification is {response.requires_clarification} and the question is {response.question} and the verification message is {response.verification_message}')

    # if we need clarification go to END state and add messages AIMessage with clarification question 
    if response.requires_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # if we have enough clarity, proceed with the write_research_brief
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification_message)]}
        )


def write_research_brief(state: AgentState):

    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """

    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    # pass all the messages to the LLM - get_buffer_string
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}.")]
    }

