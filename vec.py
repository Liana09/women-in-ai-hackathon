# very useful: https://github.com/stefanwebb/odsc-ai-builders-summit-2025/blob/main/notebooks/3%20building%20a%20rag%20system%20with%20milvus%20-%20apple%20silicon.ipynb
# This was what we were recommended  https://github.com/stefanwebb/odsc-ai-builders-summit-2025/tree/main/notebooks 
# Create/Load a Milvus collection (like in the quickstart). Read your local  files from midnighthome/Builds/womeindatahackathon/california_report.json Extract text (using PyPDF2). Embed the extracted text with DefaultEmbeddingFunction. Insert the embeddings and metadata into the Milvus collection.
#importing

import os
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, MilvusClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import JSONLoader  # Updated import
import json

# Initialize Milvus client
client = MilvusClient("my-milvus.db")

# Define Milvus collection schema
schema = CollectionSchema(
    fields=[
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=768),
        FieldSchema("text", DataType.VARCHAR, max_length=65535)
    ]
)

# Path to the JSON file
json_path = "california_report.json"

# Define a JQ query string based on your JSON structure
jq_query = '.lines[]'  # Adjusted to match the "lines" array

# Initialize JSONLoader with the correct JQ query
loader = JSONLoader(json_path, jq_schema=jq_query)
print("loader", loader)
try:
    documents = loader.load()
    texts = [doc.page_content for doc in documents]
except json.JSONDecodeError as e:
    print(f"JSON Decode Error: {e}")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit(1)

# Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# Check and manage Milvus collection
collection_name = "json_collection"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)

# Create a new Milvus collection
collection = client.create_collection(
    collection_name=collection_name,
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

data = [
    {"id": i, "vector": embeddings[i], "text": texts[i], "subject": "history"}
    for i in range(len(embeddings))
]
print("texts",texts)

# Insert embeddings and texts into the collection
client.insert(
    collection_name=collection_name,
    data=data
)


embeddings2 = model.encode("This is a dummy text")
data2 = [
    {"id": i, "vector": embeddings2[i], "text": "This is a dummy text", "subject": "history"}
    for i in range(len(embeddings2))
]

search_results = client.search(
    collection_name=collection_name,
    data=["sdfkjdsnkjfsnfjks"],
    search_params={"metric_type": "COSINE", "params": {}},  # Search parameters
)[0]
#not useful - just mentions JSON https://milvus.io/docs/import-data.md and looking at this: https://docs.unstructured.io/open-source/core-functionality/embedding