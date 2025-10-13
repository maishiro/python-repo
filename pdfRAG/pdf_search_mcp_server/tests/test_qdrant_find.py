import asyncio
import unittest
from unittest.mock import patch, MagicMock
import main


class TestQdrantFind(unittest.TestCase):
    def test_returns_contents_and_calls_similarity_search_with_k_default(self):
        query = "sample query"

        with patch("main.OllamaEmbeddings") as mock_emb, \
             patch("main.QdrantClient") as mock_client, \
             patch("main.QdrantVectorStore") as mock_store:

            # Setup mocks
            mock_emb.return_value = MagicMock()
            mock_client.return_value = MagicMock()

            docs = [MagicMock(page_content="doc1"), MagicMock(page_content="doc2")]
            store_instance = MagicMock()
            store_instance.similarity_search.return_value = docs
            mock_store.return_value = store_instance

            # Run the async function
            result = asyncio.run(main.qdrant_find(query))

            # Assert returned content matches mocked docs (joined by newline)
            self.assertEqual(result, "doc1\ndoc2")

            # Ensure similarity_search was called with the query and default k=5
            store_instance.similarity_search.assert_called_with(query, k=5)

    def test_handles_no_results(self):
        query = "no results"

        with patch("main.OllamaEmbeddings") as mock_emb, \
             patch("main.QdrantClient") as mock_client, \
             patch("main.QdrantVectorStore") as mock_store:

            mock_emb.return_value = MagicMock()
            mock_client.return_value = MagicMock()

            store_instance = MagicMock()
            store_instance.similarity_search.return_value = []
            mock_store.return_value = store_instance

            result = asyncio.run(main.qdrant_find(query))

            # When no documents are found, implementation returns empty string
            self.assertEqual(result, "")
            store_instance.similarity_search.assert_called_with(query, k=5)


if __name__ == "__main__":
    unittest.main()
