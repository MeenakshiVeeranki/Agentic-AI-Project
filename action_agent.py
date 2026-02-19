import json
import re
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def _safe_json(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        return json.loads(match.group()) if match else []

def action_agent(state):
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 6})

    docs = retriever.invoke(
        "tasks, action items, responsibilities, dependencies, deadlines"
    )

    text = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are an Action & Dependency Extraction Agent.

Extract ONLY real, executable action items.
If the document is descriptive or historical and contains NO actions,
return an empty list [].

Return STRICT JSON only.

Format:
[
  {{
    "task": "<what needs to be done>",
    "owner": "<person/team or 'unknown'>",
    "dependency": "<blocking task or 'none'>",
    "deadline": "<date or 'unknown'>"
  }}
]

Rules:
- Do NOT hallucinate actions
- If nothing actionable exists, return []
- No explanations, no markdown

TEXT:
{text}
"""

    response = llm.invoke(prompt).content
    actions = _safe_json(response)

    return {"actions": actions}
