# Face detection with photo capture after 3 seconds of detection

import cv2 as cv
import os
import time
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

urlPath = "http://192.168.1.16:8009/recognition/upload/"
file_path = ""

# INITIALIZE
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

# Variables to track detection time
detection_start_time = None
detection_threshold = 3  # seconds
saved_image = False

# WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    if len(faces) > 0:
        # If a face is detected
        x, y, w, h = faces[0]  # Consider the first detected face

        # Draw rectangle on the frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Start or check the detection timer
        if detection_start_time is None:
            detection_start_time = time.time()
        else:
            elapsed_time = time.time() - detection_start_time
            if elapsed_time > detection_threshold and not saved_image:
                # Save the photo after threshold timeIt f
                photo_path = f"detected_face.jpg"

                cv.imwrite(photo_path, frame)

                print(f"Photo saved: {photo_path}")

                saved_image = True

                with open(photo_path, "rb") as file:
                    files = {"file": file}
                    response = requests.post(urlPath, files=files)

                if response.status_code == 200:
                    print("Resultado:", response.json())
                else:
                    print("Error:", response.status_code, response.text)
    else:
        # Reset detection timer if no face is detected
        detection_start_time = None
        saved_image = False

    cv.imshow("Face Detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
