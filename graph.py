from langgraph.graph import StateGraph, END
from state import GraphState

from agents.summary_agent import summary_agent
from agents.action_agent import action_agent
from agents.risk_agent import risk_agent
from utils.vectorstore import build_vector_store

def orchestrator_init(state: GraphState) -> GraphState:
    state["memory"] = {
        "doc_purpose": "",
        "key_entities": [],
        "decisions": [],
        "constraints": []
    }
    state["vectorstore"] = build_vector_store(state["chunks"])
    return state

def orchestrator_update_memory(state: GraphState) -> GraphState:
    state["memory"]["doc_purpose"] = state["summary"]
    return state

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator_init", orchestrator_init)
    graph.add_node("summary_agent", summary_agent)
    graph.add_node("orchestrator_update", orchestrator_update_memory)
    graph.add_node("action_agent", action_agent)
    graph.add_node("risk_agent", risk_agent)

    graph.set_entry_point("orchestrator_init")
    graph.add_edge("orchestrator_init", "summary_agent")
    graph.add_edge("summary_agent", "orchestrator_update")
    graph.add_edge("orchestrator_update", "action_agent")
    graph.add_edge("orchestrator_update", "risk_agent")
    graph.add_edge("action_agent", END)
    graph.add_edge("risk_agent", END)

    return graph.compile()
