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

def risk_agent(state):
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 6})

    docs = retriever.invoke(
        "unresolved questions, ambiguities, missing information, limitations, risks"
    )

    text = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are a Risk & Open-Issues Agent.

Identify ONLY:
- unresolved questions
- missing information
- ambiguities
- potential risks or limitations

If the document is purely informational and has NO risks or open issues,
return an empty list [].

Return STRICT JSON only.

Format:
[
  {{
    "type": "<Missing Info | Assumption | Risk | Open Question>",
    "description": "<what is missing or unclear>",
    "impact": "<why it matters>"
  }}
]

Rules:
- Do NOT summarize history
- Do NOT restate known facts
- No hallucination
- No explanations

TEXT:
{text}
"""

    response = llm.invoke(prompt).content
    risks = _safe_json(response)

    return {"risks": risks}
