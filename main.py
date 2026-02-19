import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from graph import build_graph
from utils.chunker import chunk_text

def load_document(relative_path: str) -> str:
    path = os.path.join(BASE_DIR, relative_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    text = load_document("data/input.txt")
    chunks = chunk_text(text)

    state = {
        "chunks": chunks,
        "memory": {},
        "vectorstore": None,
        "summary": "",
        "actions": [],
        "risks": []
    }

    app = build_graph()
    result = app.invoke(state)

    print("\n===== SUMMARY =====\n")
    print(result["summary"])

    print("\n===== ACTION ITEMS =====\n")
    for a in result["actions"]:
        print(a)

    print("\n===== RISKS & OPEN ISSUES =====\n")
    for r in result["risks"]:
        print(r)

if __name__ == "__main__":
    main()
