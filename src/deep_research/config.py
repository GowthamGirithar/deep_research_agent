"""
Configuration module for Deep Research Agent.

This module provides centralized configuration management for all agents,
reading model configurations from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for Deep Research Agent."""
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model configurations for different agents
    SCOPE_AGENT_MODEL = os.getenv("SCOPE_AGENT_MODEL", "openai:gpt-4.1")
    RESEARCH_AGENT_MODEL = os.getenv("RESEARCH_AGENT_MODEL", "openai:gpt-4.1")
    SUPERVISOR_AGENT_MODEL = os.getenv("SUPERVISOR_AGENT_MODEL", "openai:gpt-4.1")
    FINAL_REPORT_AGENT_MODEL = os.getenv("FINAL_REPORT_AGENT_MODEL", "openai:gpt-4.1")
    
    @classmethod
    def get_scope_agent_model(cls) -> str:
        """Get the model configuration for the scope agent."""
        return cls.SCOPE_AGENT_MODEL
    
    @classmethod
    def get_research_agent_model(cls) -> str:
        """Get the model configuration for the research agent."""
        return cls.RESEARCH_AGENT_MODEL
    
    @classmethod
    def get_supervisor_agent_model(cls) -> str:
        """Get the model configuration for the supervisor agent."""
        return cls.SUPERVISOR_AGENT_MODEL
    
    @classmethod
    def get_final_report_agent_model(cls) -> str:
        """Get the model configuration for the final report agent."""
        return cls.FINAL_REPORT_AGENT_MODEL

# Create a global config instance
config = Config()