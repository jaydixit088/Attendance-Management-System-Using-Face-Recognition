import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

# Path to the directory containing your training images
path = "C:\\Project\\Training images"
images = []
classNames = []

# Authentication data for admin
admin_auth_data = {"admin": "adminpassword"}
admin_logged_in = False

# Function to authenticate admin


def admin_login():
    global admin_logged_in
    if admin_logged_in:
        return True
    username = input("Enter admin username: ")
    password = input("Enter admin password: ")
    if admin_auth_data.get(username) == password:
        admin_logged_in = True
        return True
    else:
        return False

# Function to mark attendance


def markAttendance(name, present):
    with open("C:\\Project\\Attendance.csv", 'a') as f:
        if present:
            f.write(f'{name},{datetime.now()},Present\n')
        else:
            f.write(f'{name},{datetime.now()},Absent\n')

# Function to check elapsed time


def check_elapsed_time(start_time):
    elapsed_time = time.time() - start_time
    return elapsed_time >= 15 * 60  # 15 minutes


# Authenticate admin before proceeding
if not admin_login():
    exit()

# Ensuring error handling for image loading
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Error loading image: {cl}")

print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError as e:
            print(f"Encoding not found in {images.index(img)}. Error: {e}")
    return encodeList


encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Use the correct camera index. If your webcam isn't working, try changing the 0 to 1, 2, etc.
cap = cv2.VideoCapture(0)

start_time = time.time()  # Record start time

# Dictionary to store the last seen time for each student
last_seen = {name: start_time for name in classNames}

# Set to keep track of students whose attendance has already been marked
marked_students = set()

while True:
    success, img = cap.read()
    if not success:
        break

    # Check if 15 minutes have elapsed
    if check_elapsed_time(start_time):
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    present_students = set()  # Set to store recognized student names

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(
            encodeListKnown, encodeFace, tolerance=0.50)
        if True in matches:
            name = classNames[matches.index(True)]
            present_students.add(name)
            last_seen[name] = time.time()
            if name not in marked_students:
                # Draw rectangle around the face and put text
                top, right, bottom, left = faceLoc
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(img, (left, top),
                              (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name, True)  # Mark as present
                marked_students.add(name)

    # Mark students who are not recognized as absent after 15 minutes
    for name, last_seen_time in last_seen.items():
        if time.time() - last_seen_time >= 15 * 60 and name not in present_students and name not in marked_students:
            markAttendance(name, False)  # Mark as absent
            marked_students.add(name)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
