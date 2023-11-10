import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from pinecone_client import query

test_image = face_recognition.load_image_file("../images/test/santi_test_1.jpg")

# Gives the 128-dimension face encoding for each face in the image
test_encodings = face_recognition.face_encodings(test_image)
test_locations = face_recognition.face_locations(test_image)

# Fetches the predominant face in the image
data = query(test_encodings[0].tolist(), top_k=1, include_metadata=True, include_values=True)

# Create an array with the know (pinecone) face encodings of the faces in the image
known_face_encodings = [data.matches[0].values]

# Create an array with the names of the faces in the image
known_face_names = [data.matches[0].metadata["name"]]

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
pil_image = Image.fromarray(test_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the unknown image
for (top, right, bottom, left), face_encoding in zip(test_locations, test_encodings):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

    name = "Unknown"

    # Or instead, use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    # Draw a box around the face using the Pillow module
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # Draw a label with a name below the face
    _, _, text_width, text_height = draw.textbbox((left, bottom - 300), name, font=None)
    draw.text((left + 6, bottom - text_height - 150), name, fill=(255, 255, 255, 255))


# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()