import face_recognition
import uuid
from pinecone_client import upsert

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("../images/data/santi.jpeg")
# Gives the 128-dimension face encoding for each face in the image
encoding = face_recognition.face_encodings(image)[0].tolist()

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

upsert(vectors)