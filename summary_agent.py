from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def summary_agent(state):
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(
        "overall purpose, intent, constraints, and key decisions"
    )

    text = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are a Context-Aware Summary Agent.

TEXT:
{text}

Return a concise summary only.
"""

    summary = llm.invoke(prompt).content

    # 🔑 return ONLY what this agent updates
    return {"summary": summary}
