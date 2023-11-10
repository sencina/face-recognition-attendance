# face-recognition-attendance
Final project for subject "Visión Artificial" (Artificial Vision)

## Steps to setup the code
Create a .envrc file and copy the .envrc.template content and complete it with your own variables
run direnv allow
run pip install -r requirements.txt

you are all done to run the project!

## Steps to run the project
run cd scripts
run python server.py

## Endpoints
GET /health
check if the server is running (200)

POST /upload/<name>
Body: image in binary data
Params: Name= name of the student
Uploads an image, vectorizes it, saves it in the vector db and returns a 201 if it saved correctly

POST /attendance
Body: image in binary data
Detects the face location and matches the face encodings of the image with the ones in the databse, returns a list of names of the students who attended to class (200)
