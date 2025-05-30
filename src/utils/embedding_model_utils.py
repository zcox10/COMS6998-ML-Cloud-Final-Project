import logging
import uuid
from typing import List
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.utils.yaml_parser import YamlParser
from src.utils.secrets_utils import SecretsUtils


class EmbeddingModelUtils:
    """
    Utility class to handle embedding model initialization and embedding storage
    in a Qdrant vector database. Loads configuration, sets up the embedding
    model, manages Qdrant collection creation, and provides a method to upsert
    document embeddings in an idempotent way.
    """

    def __init__(self, vector_db_url):
        """
        Initialize the embedding utils:
        - Load YAML configuration to get Qdrant collection name
        - Initialize the embedding model
        - Determine embedding vector size
        - Connect to Qdrant and (re)create the target collection
        """

        # Load config
        self._config = YamlParser("./config.yaml")
        self._collection_name = self._config.get_field("gcp.gke.services.vector_db.collection_name")

        # Embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=SecretsUtils().get_secret(secret_id="GOOGLE_GEMINI_KEY"),
        )
        self.embedding_size = self._get_embedding_size()

        # Vector db
        self.vector_db_client = QdrantClient(url=vector_db_url, prefer_grpc=True)

    def recreate_vector_db_collection(self):
        self.vector_db_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
        )

    def retrieve_context(self, question: str, top_k: int) -> str:
        # embed user's question
        query_vector = self.embedding_model.embed_query(question)

        # retrieve top-k hits
        hits = self.vector_db_client.search(
            collection_name=self._collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        # pull chunked text from each hit's payload and generate context string
        snippets = [hit.payload["text"] for hit in hits]
        return "\n\n".join(snippets)

    def reset_collection(self):
        self.vector_db_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
        )
        logging.info(f"Reset {self._collection_name} collection")
        self.get_collection_point_count()

    def get_collection_point_count(self):
        result = self.vector_db_client.count(
            collection_name=self._collection_name,
        )
        logging.info(f"{result.count} total points in {self._collection_name}")

    def upsert_document_embedding(self, docs: List[Document], entry_id: str) -> None:
        """
        Compute embeddings for a list of document chunks and upsert them into
        the Qdrant collection. Uses deterministic UUID based on entry_id and
        chunk index to ensure idempotency (same IDs overwrite previous points).

        Args:
            docs (List[Any]): A list of chunked documents, each with
                attributes `.page_content` (str) and `.metadata` (dict).
            entry_id (str): Unique identifier for the source document/paper.
        """

        # Extract text contents and batch-embed them
        texts = [d.page_content for d in docs]
        vectors = self.embedding_model.embed_documents(texts)

        # Build idempotent upsert payload
        points = []
        for i, (vec, doc) in enumerate(zip(vectors, docs)):

            # generate point uuid each entry_id for reproducibility
            point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"{entry_id}-{i}")
            payload = {"text": doc.page_content, **doc.metadata}
            points.append(PointStruct(id=str(point_uuid), vector=vec, payload=payload))

        # Upsert into Qdrant; same IDs overwrite old points
        self.vector_db_client.upsert(collection_name=self._collection_name, points=points)
        logging.info(f"Upserted {len(points)} chunks for {entry_id}")

    def _get_embedding_size(self):
        return len(self.embedding_model.embed_query("test"))

    def debug_print_points(self, limit: int = 10) -> None:
        """
        Fetch up to `limit` points from the collection and print out
        their payloads (including the 'text' field).
        """
        # scroll through the first `limit` points
        points = self.vector_db_client.scroll(
            collection_name=self._collection_name, limit=limit, with_payload=True
        )
        for pt in points:
            print(pt)
            # print(f"Point ID: {pt.id}")
            # # this should show your chunk text
            # print(" text:", pt.payload.get("text"))
            # # and any other metadata
            # others = {k: v for k, v in pt.payload.items() if k != "text"}
            # print(" metadata:", others)
            # print("---")
