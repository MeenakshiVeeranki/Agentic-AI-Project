from langchain_openai import ChatOpenAI
from state import GraphState

# LLM configuration
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

def summary_agent(state: GraphState) -> GraphState:
    """
    Context-Aware Summary Agent
    """
    chunk_summaries = []

    for chunk in state["chunks"]:
        prompt = f"""
You are a Context-Aware Summary Agent.

Summarize the following text while preserving:
- intent
- constraints
- decisions

TEXT:
{chunk['text']}

Return a concise summary only.
"""
        response = llm.invoke(prompt)
        chunk_summaries.append(response.content)

    merge_prompt = f"""
Merge the following summaries into ONE coherent summary.
Avoid repetition. Preserve key decisions.

Summaries:
{chunk_summaries}
"""

    final_summary = llm.invoke(merge_prompt).content
    state["summary"] = final_summary

    return state
