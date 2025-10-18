from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.docstore.document import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm
import pymupdf
import glob
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
# Vector DB
VECTOR_DB_URL = "http://localhost:6333"
model_name = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_COLLECTION = "demo_collection"
CHUNK_SIZE = 256  # チャンクサイズ
EMBED_DIM = 1024  # 埋め込み次元数

def process_pdf(pdf_path):
    try:
        docs = pymupdf.open(pdf_path)
        pdf_docs = []
        total_pages = docs.page_count  # ページ数を取得
        logger.info(f"{pdf_path}: 全{total_pages}ページを処理開始")

        for page_num in tqdm(range(total_pages), desc=f"Processing {pdf_path}", ncols=100):
            try:
                page = docs[page_num]
                text = page.get_text()
                pdf_docs.append(Document(page_content=text))
            except Exception as e:
                logger.error(f"ページ処理エラー ({pdf_path} - {page_num + 1}ページ目): {e}")
        return pdf_docs
    except Exception as e:
        logger.error(f"PDF読み込みエラー ({pdf_path}): {e}")
        return []
    finally:
        if 'docs' in locals():
            docs.close()

def batch_iterator(iterable, batch_size):
    """指定したサイズでバッチに分割するイテレータ"""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def ingest_pdfs(batch_size=10):
    pdf_files = sorted(glob.glob("pdfs/*.pdf"))
    all_docs = []
    failed_pdfs = []

    # PDFを1つずつ処理
    for pdf in tqdm(pdf_files, desc="PDFファイル処理中", ncols=100):
        try:
            result = process_pdf(pdf)
            if result:
                all_docs.extend(result)
                logger.info(f"完了: {pdf} - {len(result)}ページを処理しました")
            else:
                failed_pdfs.append(pdf)
                logger.error(f"失敗: {pdf} - 読み込みエラー")
        except Exception as e:
            logger.error(f"エラー ({pdf}): {str(e)}")
            failed_pdfs.append(pdf)

    # 未処理のファイルがある場合は表示
    remaining_pdfs = set(pdf_files) - set(p for p in pdf_files if p not in failed_pdfs)
    if remaining_pdfs:
        logger.info("未処理のファイル:")
        for pdf in remaining_pdfs:
            logger.info(f"- {pdf}")

    # 処理結果の詳細表示
    logger.info("===== PDF処理結果 =====")
    logger.info(f"処理対象: {len(pdf_files)}ファイル")
    logger.info(f"成功: {len(pdf_files) - len(failed_pdfs)}ファイル")
    logger.info(f"失敗: {len(failed_pdfs)}ファイル")
    if failed_pdfs:
        logger.info("失敗したファイル:")
        for pdf in failed_pdfs:
            logger.info(f"- {pdf}")
    logger.info(f"抽出ページ数: {len(all_docs)}ページ")
    logger.info("=====================")

    # embeddings の初期化を先に行い、次元数を取得
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    embed_dim = embeddings.embedding_size if hasattr(embeddings, "embedding_size") else EMBED_DIM

    # チャンク分割を最適化（小さめのチャンクサイズを使用）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=50,  # オーバーラップも調整
        add_start_index=True,
        length_function=len
    )
    chunks = splitter.split_documents(all_docs)

    # client = QdrantClient(":memory:")
    # client = QdrantClient(path="path/to/db")
    # client = QdrantClient(host="localhost", port=6333)
    client = QdrantClient(url=VECTOR_DB_URL)

    try:
        client.delete_collection(collection_name=VECTOR_DB_COLLECTION)
    except Exception as e:
        logger.error(f"コレクション削除時の例外 (無視して継続): {e}")        
    client.create_collection(
        VECTOR_DB_COLLECTION,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )

    # ベクトル登録（バッチ処理）
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=VECTOR_DB_COLLECTION,
        embedding=embeddings,
    )

    total_chunks = len(chunks)
    success_count = 0
    logger.info(f"全チャンク数: {total_chunks}")

    try:
        failed_chunks = []
        # すべてのチャンクをバッチ処理
        for i, batch in enumerate(tqdm(batch_iterator(chunks, batch_size),
                         desc="ベクトル登録中",
                         total=(total_chunks + batch_size - 1) // batch_size)):
            try:
                # バッチ内の各チャンクを個別に処理
                for chunk in batch:
                    try:
                        vector_store.add_documents([chunk])
                        success_count += 1
                    except Exception as chunk_error:
                        logger.error(f"チャンク処理エラー: {str(chunk_error)[:100]}...")
                        failed_chunks.append(chunk)

                if (i + 1) % 5 == 0:
                    logger.info(f"進捗: {success_count}/{total_chunks} チャンク完了")
            except Exception as batch_error:
                logger.error(f"バッチ {i+1} の処理中にエラー発生: {str(batch_error)[:100]}...")
                continue

        # 登録完了後の確認
        try:
            collection_info = client.get_collection(VECTOR_DB_COLLECTION)
            actual_vectors = collection_info.vectors_count if collection_info else 0
        except Exception as e:
            logger.error(f"コレクション情報取得エラー: {e}")
            actual_vectors = success_count

        logger.info("登録完了:")
        logger.info(f"処理したチャンク数: {total_chunks}")
        logger.info(f"成功したチャンク数: {success_count}")
        logger.info(f"失敗したチャンク数: {len(failed_chunks)}")
        logger.info(f"DBに登録されたベクトル数: {actual_vectors}")

        if total_chunks > 0 and success_count > 0:
            success_rate = min(100.0, (success_count / total_chunks) * 100)
            logger.info(f"成功率: {success_rate:.2f}%")
        else:
            logger.info("有効なチャンクは処理されませんでした")

    except Exception as e:
        logger.error(f"処理全体でエラーが発生: {e}")
        if success_count > 0:
            return success_count
        return 0

    return success_count

if __name__ == "__main__":
    total_processed = ingest_pdfs()
    logger.info(f"処理を完了しました。合計 {total_processed} チャンクを処理しました。")
