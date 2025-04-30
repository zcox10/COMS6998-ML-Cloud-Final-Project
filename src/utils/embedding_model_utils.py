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

    def __init__(self):
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
        self.vector_db_client = QdrantClient(
            url="qdrant.vector-db.svc.cluster.local:6333",  # kubeflow
            # url="localhost:6333",  # local
            prefer_grpc=True,
        )
        self.vector_db_client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
        )

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
            points.append(PointStruct(id=str(point_uuid), vector=vec, payload=doc.metadata))

        # Upsert into Qdrant; same IDs overwrite old points
        self.vector_db_client.upsert(collection_name=self._collection_name, points=points)
        logging.info(f"Upserted {len(points)} chunks for {entry_id}")

    def _get_embedding_size(self):
        return len(self.embedding_model.embed_query("test"))
