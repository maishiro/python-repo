# Qdrant＋LangChain-Ollamaによる日本語PDF検索システム

このプロジェクトは、日本語PDFファイルを読み込み、LangChainとOllamaでテキスト埋め込みを生成し、Qdrantにベクトル登録・保存するプログラム（プログラム1）と、Streamlitを用いてQdrantから検索し結果を表示するプログラム（プログラム2）で構成されています。

---

## 必要環境

- Python 3.8以上
- Qdrant（Docker推奨）
- Ollama（ローカルLLMサーバー、Model: `llama3.1:8b`）
- 利用ライブラリ（以下Pythonパッケージ）：
  - langchain-ollama
  - langchain-qdrant
  - langchain-text-splitters
  - langchain-community
  - pymupdf
  - qdrant-client
  - streamlit

---

## QdrantのDockerセットアップ

ローカルにQdrantをDockerで起動する手順です。

```powershell
docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 -v $pwd/qdrant_storage:/qdrant/storage qdrant/qdrant
```

- ポート6333でQdrantが起動します。

---

## プログラム1：PDF読み込み・ベクトル登録

### 概要

日本語PDFからテキストを抽出・チャンク分割し、Ollamaの埋め込みモデルでベクトル作成後、Qdrantに登録します。

### 実行例

```powershell
cd .\pdf_ingest\
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r .\requirements.txt

python ingest.py
```

### ingest_pdf_to_qdrant.pyのポイント

- PDFは`pdfs/*.pdf`に格納してください。
- Qdrantサーバーは `http://localhost:6333` へ接続します。
- Ollamaサーバーは `http://localhost:11434` へ接続します。
- 既存Qdrantコレクションは自動削除・再作成して次元数不一致を回避します。

---

## プログラム2：StreamlitによるQdrant検索インターフェース

### 概要

Streamlitを使ってQdrantに格納したベクトルを検索し、部分テキストとメタ情報を表示する簡易UIです。

### 実行例

```powershell
cd .\pdf_search\
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r .\requirements.txt

streamlit run app.py
```

### streamlit_query_qdrant.py のポイント

- LangChain Ollama埋め込みを使い、検索語からQdrantの類似文章を取得・表示します。
- Qdrant接続は`http://localhost:6333`です。
- 検索結果は最大5件まで表示されます。

---

## プログラム3：MCPによるQdrant検索提供

### 概要

Qdrantに格納したベクトルを検索し、情報を取得します。

### 環境復元例

```powershell
cd pdf_search_mcp_server
uv sync
```

### MCPサーバー設定例

```json
{
  "mcpServers": {
    "weather": {
      "command": "uv",
      "args": [
        "--directory",
        "C:\\ABSOLUTE\\PATH\\TO\\PARENT\\FOLDER\\pdf_search_mcp_server",
        "run",
        "main.py"
      ]
    }
  }
}
```

---

## 注意事項

- OllamaとQdrantサーバーはローカルで同時起動しておく必要があります。
- 埋め込みモデルの次元数とQdrantコレクションのベクトル次元数は必ず一致していなければなりません。プログラム1には自動的に一致させる処理が含まれています。
- QdrantのポートやURLを変更する場合はプログラム内の接続設定も合わせて修正してください。

---

## 参考情報

- LangChainの公式Qdrant統合ドキュメント: https://python.langchain.com/en/latest/modules/indexes/vector_stores/integrations/qdrant.html
- Qdrant公式Docker Hub: https://hub.docker.com/r/qdrant/qdrant
- Ollama公式サービス(ローカルLLMサーバー) https://ollama.com/
