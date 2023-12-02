from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw
import io
import face_recognition
import uuid
from pinecone_client import upsert, query
import tempfile
import os

TRESHOLD = 0.9

app = Flask(__name__)
CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

ANNOTATED_IMAGES_DIR = 'annotated_images'
if not os.path.exists(ANNOTATED_IMAGES_DIR):
    os.makedirs(ANNOTATED_IMAGES_DIR)

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

    # Convert NumPy array to Image object
    loaded_image_pil = Image.fromarray(loaded_image)

    draw = ImageDraw.Draw(loaded_image_pil)

    for face_location in face_locations:
        top, right, bottom, left = face_location
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

    tagged_image_buffer = io.BytesIO()
    loaded_image_pil.save(tagged_image_buffer, format='JPEG')
    tagged_image_buffer.seek(0)  # Move the cursor to the beginning of the buffer

    encoding = face_encodings[0].tolist()

    name = name.replace('-', ' ')

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

    # Return the URL of the tagged image
    tagged_image_filename = f"{uuid.uuid4()}_tagged.jpg"
    tagged_image_path = os.path.join(ANNOTATED_IMAGES_DIR, tagged_image_filename)
    loaded_image_pil.save(tagged_image_path, format='JPEG')
    tagged_image_url = f"/get_annotated_image/{tagged_image_filename}"
    return jsonify({'message': 'Face detected and tagged successfully', 'tagged_image_url': tagged_image_url}), 201


@app.route('/attendance', methods=['POST'])
def attendance():
    try:
        image_data = request.get_data()
        image = Image.open(io.BytesIO(image_data))
    except UnidentifiedImageError as e:
        return jsonify({'error': 'Invalid image format'}), 400

    temp = tempfile.NamedTemporaryFile(delete=False)
    image.save(temp, format='JPEG')

    loaded_image = face_recognition.load_image_file(temp.name)
    locations = face_recognition.face_locations(loaded_image)
    encodings = face_recognition.face_encodings(loaded_image, locations)

    attndees = []
    draw = ImageDraw.Draw(image)

    for location, encoding in zip(locations, encodings):
        top, right, bottom, left = location
        data = query(encoding.tolist(), top_k=1, include_metadata=True, include_values=True)
        print(data)

        if data.matches[0].score >= TRESHOLD:
            print(data.matches[0].metadata['name'], "MATCHED")
            name = data.matches[0].metadata['name']
            attndees.append(name)

            # Draw a square around the face
            draw.rectangle([left, top, right, bottom], outline="green", width=2)

            # Annotate the image with the name
            draw.text((left, top - 10), name, fill="green")
        else:
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

    temp.close()

    # Save the annotated image to a file
    annotated_image_filename = f"{uuid.uuid4()}_annotated.jpg"
    annotated_image_path = os.path.join(ANNOTATED_IMAGES_DIR, annotated_image_filename)
    image.save(annotated_image_path, format='JPEG')

    # Return the URL of the annotated image
    annotated_image_url = f"/get_annotated_image/{annotated_image_filename}"
    return jsonify({'attendees': attndees, 'annotated_image_url': annotated_image_url}), 200

@app.route('/get_annotated_image/<filename>')
def get_annotated_image(filename):
    return send_from_directory(ANNOTATED_IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(port=8000, debug=True)