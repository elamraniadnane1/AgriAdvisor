from qdrant_client import QdrantClient

# Replace with the correct address and port
qdrant_client = QdrantClient(url="http://localhost:6333")

# Now you can use qdrant_client to interact with the Qdrant server
collections = qdrant_client.get_collections()
print(collections)
