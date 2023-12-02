from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import face_recognition
import uuid
from pinecone_client import upsert, query
import tempfile
import os

TRESHOLD = 0.95

app = Flask(__name__)
CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/health')
def home():
    return "App is running"

@app.route('/upload/<name>', methods=['POST'])
def upload_image(name):
    image_data = request.get_data()
    image = Image.open(io.BytesIO(image_data))

    temp = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp, format='JPEG')

    loaded_image = face_recognition.load_image_file(temp.name)
    face_locations = face_recognition.face_locations(loaded_image)
    face_encodings = face_recognition.face_encodings(loaded_image, face_locations)

    if not face_encodings:
        temp.close()
        os.unlink(temp.name)
        return jsonify({'error': 'No face detected in the uploaded image'}), 400

    draw = ImageDraw.Draw(loaded_image)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

    tagged_image_buffer = io.BytesIO()
    loaded_image.save(tagged_image_buffer, format='JPEG')

    encoding = face_encodings[0].tolist()

    vectors = []
    user_id = str(uuid.uuid4())
    vectors.append({
        "id": user_id,
        "values": encoding,
        "metadata": {
            "id": user_id,
            "name": name,
        }
    })

    upsert(vectors)

    temp.close()
    os.unlink(temp.name)

    return jsonify({'message': 'Face detected and tagged successfully', 'tagged_image': tagged_image_buffer.getvalue().hex()}), 200

if __name__ == '__main__':
    app.run(port=8000, debug=True)