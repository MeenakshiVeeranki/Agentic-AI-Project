from typing import TypedDict, List, Dict, Any

class GraphState(TypedDict):
    chunks: List[Dict[str, Any]]
    memory: Dict[str, Any]
    vectorstore: Any
    summary: str
    actions: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]

