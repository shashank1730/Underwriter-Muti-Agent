from pydantic import BaseModel, Field
from typing_extensions import TypedDict, List
from langgraph.graph.message import add_messages
from typing import Annotated, Dict, Any

class State(TypedDict):
    """
    Represent the structure of the state used in graph
    """
    messages: Annotated[List, add_messages]
    current_result: Dict[str, Any]  # Store current node result
    status: str  # Current status


