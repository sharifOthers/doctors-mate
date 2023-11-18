import uuid
import weaviate
import streamlit as st

from qdrant_client import models, QdrantClient


WEAVIATE_URL = str(st.secrets["WEAVIATE_URL"])
WEAVIATE_API = str(st.secrets["WEAVIATE_API"])
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
Qdrant_URL = str(st.secrets["Qdrant_URL"])
COHERE_API_KEY = str(st.secrets["COHERE_API_KEY"])

qdrant_client = QdrantClient(
    Qdrant_URL,
    api_key = QDRANT_API_KEY
)

weviate_client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API),
    additional_headers={
        "X-Cohere-Api-Key": COHERE_API_KEY 
    }
)

class Weaviate:
    def get_similar_docs(embeddings):
        result = (
            weviate_client.query
            .get("Contents", ["idx", "source","content", "tokens"])
            .with_near_vector({
                "vector": embeddings,
                "certainty": 0.1
            })
            .with_limit(5)
            .with_additional(['certainty'])
            .do()
        )
        return result["data"]["Get"]["Contents"]


class Qdrant:
    @staticmethod
    def check_collection():
        response = qdrant_client.get_collections()
        for collection in response.collections:
            if collection.name != "log":
                qdrant_client.create_collection(
                    collection_name="log",
                    vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
                    timeout=120,
                )
            
    def sent_data(data):
        vectors = []
        vectors.append(models.PointStruct(id = str(uuid.uuid1()), vector = [1,2],
                    payload = data
                )
            )
        
        qdrant_client.upsert(
            collection_name="log",
            points = vectors
        )