import cv2
import numpy as np
import face_recognition
import os
import threading
import queue
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
import gridfs

MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
DATABASE_NAME = "face_recognition"
COLLECTION_NAME = "DetectionLogs"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
log_collection = db[COLLECTION_NAME]
fs = gridfs.GridFS(db)

# Global configurations
frame_queue = queue.Queue(maxsize=10)
directory = "ImagesAttendance"
os.makedirs(directory, exist_ok=True)
encodeListKnown = []
classNames = []

# Function to find encodings of known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if faces are detected before appending
            encodeList.append(encode[0])
    return encodeList

# Function to mark attendance in CSV
def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Function to initialize known encodings and class names
def initializeKnownFaces():
    global encodeListKnown, classNames
    images = []
    classNames = []
    myList = os.listdir(directory)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda cl: cv2.imread(f'{directory}/{cl}'), myList))
    images = [img for img in results if img is not None]
    classNames = [os.path.splitext(cl)[0] for cl in myList if cv2.imread(f'{directory}/{cl}') is not None]
    encodeListKnown = findEncodings(images)
    print(f"Initialized {len(classNames)} known faces.")

# Function to capture frames and put them into the queue
def capture_frames(cap):
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  # Remove the oldest frame if queue is full
            frame_queue.put(frame)
        else:
            print("Failed to capture frame from RTSP stream.")
            break

# Function to process frames
def process_frames():
    global encodeListKnown, classNames
    frame_count = 0 
    while True:
        if not frame_queue.empty():
            frame_count += 1  # Increment frame count
            if frame_count % 2 != 0:  # Skip processing if not every 2nd frame
                continue

            img = frame_queue.get()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for speed
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB
            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
            
            def save_image_face(file_name, face_image):
                new_file_path = os.path.join(directory, file_name)
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)  # Convert to BGR
                cv2.imwrite(new_file_path, face_image_bgr)
                initializeKnownFaces()

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                if not encodeListKnown:  # No known faces
                    print("No known faces found. Adding first face...")
                    # Extract the face region
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up
                    face_image = img[y1:y2, x1:x2]
                    save_image_face("1.jpg", face_image)
                    continue  # Skip further processing for this face

                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                
                if len(faceDis) == 0:  # Ensure faceDis is not empty
                    print("No face distances available. Skipping...")
                    continue

                matchIndex = np.argmin(faceDis)
                face_percent_value = 1 - faceDis[matchIndex]

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up

                if face_percent_value < 0.4:
                    face_image = img[y1:y2, x1:x2]  # Crop the face region
                    new_file_name = f"{len(encodeListKnown) + 1}.jpg"
                    save_image_face(new_file_name, face_image)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                if face_percent_value >= 0.5:
                    name = classNames[matchIndex].upper()
                    cv2.putText(img, f"{name} : {face_percent_value:.2}%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    markAttendance(name)
                else:
                    name = "Unknown"
                    cv2.putText(img,  f"{name} : {face_percent_value:.2}%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (46, 2, 209), 1)

                timestamp = datetime.now()
                log_detection_to_mongo(name, face_percent_value, timestamp)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def log_detection_to_mongo(name: str, confidence: float, timestamp: datetime):
    """Log face detection data to MongoDB."""
    try:
        log_entry = {
            "name": name,
            "confidence": confidence,
            "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        log_collection.insert_one(log_entry)
        print(f"Logged to MongoDB: {log_entry}")
    except Exception as e:
        print(f"Failed to log to MongoDB: {e}")


def main():
    initializeKnownFaces()
    # rtsp_url = "http://192.168.100.22:9912"
    rtsp_url = "rtsp://admin:Abc12345@192.168.100.14/streaming/channels/101"
    # rtsp_url = "rtsp://stream5:Abc12345@conic.myds.me:8005/Streaming/channels/501"
    cap = cv2.VideoCapture(rtsp_url)

    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    process_thread = threading.Thread(target=process_frames)

    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()

if __name__ == '__main__':
    main()
