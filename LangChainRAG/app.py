import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document

st.set_page_config(page_title="LangChain + Ollama RAG", layout="wide")
st.title("📚 LangChain + Ollama RAG WebUI")

# Ollama設定
EMBED_MODEL = "mxbai-embed-large"       # 埋め込み用
LLM_MODEL = "llama3.2:3b"               # 応答生成用

# ベクトルDB初期化
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# ドキュメント登録UI
uploaded_file = st.file_uploader("ドキュメントをアップロード", type=["txt"])
if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore.add_documents(chunks)
    st.success("ドキュメントを追加しました！")

# 質問UI
query = st.text_input("質問を入力してください")
if st.button("検索＆回答"):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    # プロンプト定義
    prompt = ChatPromptTemplate.from_messages([
        ("system", "次のコンテキストを使って質問に答えてください。答えが不明な場合は「わかりません」と答えてください。\n\n{context}"),
        ("human", "{input}")
    ])

    # ドキュメント結合チェーン
    doc_chain = create_stuff_documents_chain(llm, prompt)

    # RetrievalQA相当のチェーン
    qa_chain = create_retrieval_chain(retriever, doc_chain)

    result = qa_chain.invoke({"input": query})
    st.write("### 回答")
    st.write(result["answer"])
