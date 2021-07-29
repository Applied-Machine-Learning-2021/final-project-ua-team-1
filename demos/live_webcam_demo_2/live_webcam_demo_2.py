import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
A_image = face_recognition.load_image_file("images/abraham.png")
A_face_encoding = face_recognition.face_encodings(A_image)[0]
B_image = face_recognition.load_image_file("images/adarius.png")
B_face_encoding = face_recognition.face_encodings(B_image)[0]
C_image = face_recognition.load_image_file("images/alejandra.png")
C_face_encoding = face_recognition.face_encodings(C_image)[0]
D_image = face_recognition.load_image_file("images/alexis.png")
D_face_encoding = face_recognition.face_encodings(D_image)[0]
E_image = face_recognition.load_image_file("images/claudia.png")
E_face_encoding = face_recognition.face_encodings(E_image)[0]
F_image = face_recognition.load_image_file("images/danny.png")
F_face_encoding = face_recognition.face_encodings(F_image)[0]
G_image = face_recognition.load_image_file("images/elayne.png")
G_face_encoding = face_recognition.face_encodings(G_image)[0]
H_image = face_recognition.load_image_file("images/giancarlos.png")
H_face_encoding = face_recognition.face_encodings(H_image)[0]
I_image = face_recognition.load_image_file("images/greg.png")
I_face_encoding = face_recognition.face_encodings(I_image)[0]
J_image = face_recognition.load_image_file("images/jonathan.png")
J_face_encoding = face_recognition.face_encodings(J_image)[0]
K_image = face_recognition.load_image_file("images/jose.png")
K_face_encoding = face_recognition.face_encodings(K_image)[0]
L_image = face_recognition.load_image_file("images/julio.png")
L_face_encoding = face_recognition.face_encodings(L_image)[0]
M_image = face_recognition.load_image_file("images/lizbet.png")
M_face_encoding = face_recognition.face_encodings(M_image)[0]
N_image = face_recognition.load_image_file("images/maria.png")
N_face_encoding = face_recognition.face_encodings(N_image)[0]
O_image = face_recognition.load_image_file("images/marvin.png")
O_face_encoding = face_recognition.face_encodings(O_image)[0]
P_image = face_recognition.load_image_file("images/nkira.png")
P_face_encoding = face_recognition.face_encodings(P_image)[0]
Q_image = face_recognition.load_image_file("images/ron.png")
Q_face_encoding = face_recognition.face_encodings(Q_image)[0]
R_image = face_recognition.load_image_file("images/sam.png")
R_face_encoding = face_recognition.face_encodings(R_image)[0]
S_image = face_recognition.load_image_file("images/steve.png")
S_face_encoding = face_recognition.face_encodings(S_image)[0]
T_image = face_recognition.load_image_file("images/trinity.png")
T_face_encoding = face_recognition.face_encodings(T_image)[0]
U_image = face_recognition.load_image_file("images/wren.png")
U_face_encoding = face_recognition.face_encodings(U_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    A_face_encoding,
    B_face_encoding,
    C_face_encoding,
    D_face_encoding,
    E_face_encoding,
    F_face_encoding,
    G_face_encoding,
    H_face_encoding,
    I_face_encoding,
    J_face_encoding,
    K_face_encoding,
    L_face_encoding,
    M_face_encoding,
    N_face_encoding,
    O_face_encoding,
    P_face_encoding,
    Q_face_encoding,
    R_face_encoding,
    S_face_encoding,
    T_face_encoding,
    U_face_encoding
]
known_face_names = [
    'Abraham',
    "A'Darius",
    'Alejandra',
    'Alexis',
    'Claudia',
    'Danny',
    'Elayne',
    'Giancarlos',
    'Greg',
    'Jonathan',
    'Jose',
    'Julio',
    'Lizbet',
    'Maria',
    'Marvin',
    "N'Kira",
    'Ron',
    'Sam',
    'Steve',
    'Trinity',
    'Wren',
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()