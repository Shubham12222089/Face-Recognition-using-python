import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image

# Function to load and encode known faces
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Example: Add known faces
    image = face_recognition.load_image_file("C:\\Users\\shubh\\OneDrive\\Pictures\\IMG_20220501_231601_598.jpg")
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append("Shubham Jha")

    return known_face_encodings, known_face_names

def recognize_faces(image, known_face_encodings, known_face_names):
    # Convert image to numpy array
    img_array = np.array(image)

    # Find all face locations and face encodings
    face_locations = face_recognition.face_locations(img_array)
    face_encodings = face_recognition.face_encodings(img_array, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    return face_locations, face_names

def main():
    st.title("Face Recognition App")

    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()

    # Upload image or video
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])
    if uploaded_file is not None:
        # Handle image
        if uploaded_file.type in ["image/jpeg", "image/png"]:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            face_locations, face_names = recognize_faces(image, known_face_encodings, known_face_names)

            # Draw rectangles and names on the image
            img_array = np.array(image)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top, right, bottom, left = int(top), int(right), int(bottom), int(left)
                cv2.rectangle(img_array, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_array, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            st.image(img_array, caption="Recognized Faces", use_column_width=True)

        # Handle video
        elif uploaded_file.type == "video/mp4":
            st.video(uploaded_file)

if __name__ == "__main__":
    main()
