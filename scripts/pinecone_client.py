import pinecone
import os
import requests

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))
index = pinecone.Index(os.environ.get("PINECONE_INDEX_NAME"))

def upsert(vectors):
    url = os.environ.get("PINECONE_CONNECTION_URL") + "/vectors/upsert"

    payload = {"vectors": vectors}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Api-Key": os.environ.get("PINECONE_API_KEY")
    }

    response = requests.post(url, json=payload, headers=headers)
    return response

def query(vector, top_k=1, include_metadata=True, include_values=True):
    data = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=include_metadata,
        include_values=include_values
    )
    return data