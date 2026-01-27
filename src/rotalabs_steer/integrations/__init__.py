"""Framework integrations for steering vectors."""

try:
    from .langchain import SteeredAgentExecutor, SteeredChatModel, SteeredLLM, create_steered_agent
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    SteeredLLM = None
    SteeredChatModel = None
    SteeredAgentExecutor = None
    create_steered_agent = None

__all__ = ["SteeredLLM", "SteeredChatModel", "SteeredAgentExecutor", "create_steered_agent", "HAS_LANGCHAIN"]
