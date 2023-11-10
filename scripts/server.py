from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import io
import face_recognition
import uuid
from pinecone_client import upsert, query
import tempfile
import os


app = Flask(__name__)
CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/health')
def home():
    return "App is running"

@app.route('/upload/<name>', methods=['POST'])
def upload_image(name):
    image_data = request.get_data() # Get the image data
    image = Image.open(io.BytesIO(image_data)) # Open the image

    # Save the image to a temporary file
    temp = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp, format='JPEG')

    # Load the image file from the temporary file
    loaded_image = face_recognition.load_image_file(temp.name)
    encoding = face_recognition.face_encodings(loaded_image)[0].tolist()
    name_content = name.split('-')
    name = name_content[0] + ' ' + name_content[1]
    vectors = []
    id = str(uuid.uuid4())
    vectors.append({
        "id": id,
        "values": encoding,
        "metadata": {
            "id": id,
            "name": name,
            }
        })
    
    upsert(vectors)
    temp.close()
    os.unlink(temp.name)
    return f"Image {name} saved.", 201

@app.route('/attendance', methods=['POST'])
def attendance():
    image_data = request.get_data()
    image = Image.open(io.BytesIO(image_data))
    temp = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp, format='JPEG')
    loaded_image = face_recognition.load_image_file(temp.name)
    locations = face_recognition.face_locations(loaded_image)
    encodings = face_recognition.face_encodings(loaded_image, locations)

    attndees = []
    for encoding in encodings:
        data = query(encoding.tolist(), top_k=1, include_metadata=True, include_values=True)
        print(data)
        if data.matches[0].score >= 0.9:
            attndees.append(data.matches[0].metadata['name'])

    temp.close()
    os.unlink(temp.name)
    return attndees, 201

if __name__ == '__main__':
    app.run(port=8000,debug=True)