import streamlit as st
from langchain_community.vectorstores import Qdrant
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

import logging
from logging.handlers import RotatingFileHandler

# ロガー作成
logger = logging.getLogger("ingest")
logger.setLevel(logging.DEBUG)
# コンソール出力用ハンドラ
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# ファイル出力用ハンドラ
file_handler = RotatingFileHandler("log.txt", maxBytes=1024*1024, backupCount=5, encoding="utf-8")
# フォーマット設定
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
# ハンドラをロガーに追加
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Ollama設定
EMBED_MODEL = "mxbai-embed-large"       # 埋め込み用
LLM_MODEL = "llama3.1:8b"               # LLM用
# Vector DB
VECTOR_DB_URL = "http://localhost:6333"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_COLLECTION = "demo_collection"


st.title("PDF検索アプリ (Python + Streamlit)")

query = st.text_input("質問を入力してください:")

if query:
    # client = QdrantClient(":memory:")
    # client = QdrantClient(path="path/to/db")
    # client = QdrantClient(host="localhost", port=6333)
    client = QdrantClient(url=VECTOR_DB_URL)

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = QdrantVectorStore(client=client, collection_name=VECTOR_DB_COLLECTION, embedding=embeddings)
    docs = vector_store.similarity_search(query, k=5)

    for i, doc in enumerate(docs):
        logger.info(f"{i+1}. {doc.page_content[:200]}")
        logger.info(f"メタ情報: {doc.metadata}")

    context = "\n".join([d.page_content for d in docs])
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    prompt = f"以下の仕様を参考に答えてください:\n{context}\n\n質問: {query}"
    answer = llm.invoke(prompt)

    st.subheader("回答")
    st.write(answer)
    logger.info(f"回答: {answer}")
