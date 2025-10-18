from mcp.server.fastmcp import FastMCP
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import logging


# Initialize FastMCP server
mcp = FastMCP("document")
logging.info("FastMCP server initialized.")

# Ollama設定
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"       # 埋め込み用
LLM_MODEL = "llama3.1:8b"               # LLM用
# Vector DB
VECTOR_DB_URL = "http://localhost:6333"
VECTOR_DB_COLLECTION = "demo_collection"


# MCPツールの定義
@mcp.tool()
async def qdrant_find(query: str) -> str:
    """
    ベクトルDBから類似文書を検索します。

    Args:
        query (str): 検索クエリ
    """
    k = 5  #    k (int, optional): 取得する文書の数. デフォルトは5.
    logging.info(f"qdrant_find called with query: {query}, k: {k}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    qdrant_client = QdrantClient(VECTOR_DB_URL)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=VECTOR_DB_COLLECTION, embedding=embeddings)
    docs = vector_store.similarity_search(query, k=k)
    logging.info(f"Found {len(docs)} documents")
    return '\n'.join([doc.page_content for doc in docs])


def main():
    # Initialize and run the server
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
