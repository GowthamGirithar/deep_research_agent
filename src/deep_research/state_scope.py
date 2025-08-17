from pydantic import BaseModel, Field

# schema for the workflow

# ClarifyWithUser is to clarify with user to know full context
class ClarifyWithUser(BaseModel):
    """ for user input clarification """

    requires_clarification: bool = Field(description= "whether the user needs to be asked with clarification questions")
    
    question : str = Field(description="A question to ask the user to clarify the report scope", default=None)

    verification_message: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


# ResearchQuestion has the research_brief that can be sent to the research agent
class ResearchQuestion(BaseModel):
    """ Schema for structured research brief generation. """

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )