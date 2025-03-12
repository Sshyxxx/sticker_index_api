import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

# Initialize the client
client = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")

client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)

vectors = np.random.rand(100, 100)
# NOTE: consider splitting the data into chunks to avoid hitting the server's payload size limit
# or use `upload_collection` or `upload_points` methods which handle this for you
# WARNING: uploading points one-by-one is not recommended due to requests overhead
client.upsert(
    collection_name="my_collection",
    points=[
        PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
        )
        for idx, vector in enumerate(vectors)
    ]
)

query_vector = np.random.rand(100)
hits = client.query_points(
    collection_name="my_collection",
    query=query_vector,
    limit=5  # Return 5 closest points
)
