import face_recognition
import pinecone
import os
import uuid
import requests

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("../images/data/santi.jpeg")
# Gives the 128-dimension face encoding for each face in the image
encoding = face_recognition.face_encodings(image)[0].tolist()

# Initialize Pinecone
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIRONMENT"))

# Create the payload
vectors = []
id = str(uuid.uuid4())
vectors.append({
    "id": id,
    "values": encoding,
    "metadata": {
        "id": id,
        "name": "Santiago Encina",
        }
    })

# Send the payload to Pinecone
url = os.environ.get("PINECONE_CONNECTION_URL") + "/vectors/upsert"

payload = {"vectors": vectors}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Api-Key": os.environ.get("PINECONE_API_KEY")
}

response = requests.post(url, json=payload, headers=headers)