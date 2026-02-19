from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(chunks):
    texts = [c["text"] for c in chunks]
    metadatas = [{"chunk_id": c["id"]} for c in chunks]

    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )
