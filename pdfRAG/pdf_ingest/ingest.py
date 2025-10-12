from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pymupdf
import glob

# Ollama設定
EMBED_MODEL = "mxbai-embed-large"       # 埋め込み用
# Vector DB
VECTOR_DB_URL = "http://localhost:6333"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_COLLECTION = "demo_collection"
EMBED_DIM = 1024  # 埋め込み次元数

def ingest_pdfs():
    pdf_files = glob.glob("pdfs/*.pdf")
    all_docs  = []
    for pdf in pdf_files:
        docs = pymupdf.open(pdf)
        for page in docs: # iterate the document pages
            print(f"Loaded page {page.number} from {pdf}")
            text = page.get_text()
            print(text[:500])
            doc = [Document(page_content=text)]
            all_docs.extend(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=EMBED_DIM, chunk_overlap=200, add_start_index=True)
    chunks = splitter.split_documents(all_docs)

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    embed_dim = embeddings.embedding_size if hasattr(embeddings, "embedding_size") else EMBED_DIM


    # client = QdrantClient(":memory:")
    # client = QdrantClient(path="path/to/db")
    # client = QdrantClient(host="localhost", port=6333)
    client = QdrantClient(url=VECTOR_DB_URL)

    try:
        client.delete_collection(collection_name=VECTOR_DB_COLLECTION)
    except Exception as e:
        print(f"コレクション削除時の例外 (無視して継続): {e}")        
    client.create_collection(
        VECTOR_DB_COLLECTION,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )

    # ベクトル登録
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=VECTOR_DB_URL,          # ここにQdrantサーバーURLを指定
        collection_name=VECTOR_DB_COLLECTION,
        force_recreate=True         # 既存コレクションがあれば削除して再作成
    )
    vector_store.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks into Qdrant.")

if __name__ == "__main__":
    ingest_pdfs()
