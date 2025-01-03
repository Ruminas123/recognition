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

MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "face_recognition"
COLLECTION_NAME = "DetectionLogs"

# ตั้งค่าการเชื่อมต่อ MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
log_collection = db[COLLECTION_NAME]
fs = gridfs.GridFS(db)

# การตั้งค่าทั่วไป
frame_queue = queue.Queue(maxsize=10)
directory = "ImagesAttendance"
os.makedirs(directory, exist_ok=True)
encodeListKnown = []
classNames = []
stop_event = threading.Event()  # ใช้สำหรับหยุด Threads

# ฟังก์ชันสำหรับหาการเข้ารหัสใบหน้าที่รู้จัก
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

# ฟังก์ชันสำหรับบันทึกการเข้าร่วม
def markAttendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Time\n")
    with open("attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# ฟังก์ชันสำหรับบันทึกข้อมูลลง MongoDB
def log_detection_to_mongo(name, confidence, timestamp):
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

# ฟังก์ชันสำหรับเตรียมใบหน้าที่รู้จัก
def initializeKnownFaces():
    global encodeListKnown, classNames
    images = []
    myList = os.listdir(directory)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda cl: cv2.imread(f'{directory}/{cl}'), myList))
    images = [img for img in results if img is not None]
    classNames = [os.path.splitext(cl)[0] for cl in myList if cv2.imread(f'{directory}/{cl}') is not None]
    encodeListKnown = findEncodings(images)
    print(f"Initialized {len(classNames)} known faces.")

# ฟังก์ชันสำหรับจับภาพ
def capture_frames(cap):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put(frame)

# ฟังก์ชันสำหรับประมวลผลเฟรม
def process_frames():
    global encodeListKnown, classNames
    while not stop_event.is_set():
        if not frame_queue.empty():
            img = frame_queue.get()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                if not encodeListKnown:
                    face_image = img[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(directory, "1.jpg"), face_image)
                    initializeKnownFaces()
                    continue

                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                face_percent_value = 1 - faceDis[matchIndex]

                if face_percent_value >= 0.5:
                    name = classNames[matchIndex].upper()
                    markAttendance(name)
                else:
                    name = "Unknown"

                log_detection_to_mongo(name, face_percent_value, datetime.now())

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

# ฟังก์ชันหลัก
def main():
    initializeKnownFaces()
    rtsp_url = "rtsp://admin:Abc12345@192.168.100.14/streaming/channels/101"
    cap = cv2.VideoCapture(rtsp_url)

    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    process_thread = threading.Thread(target=process_frames)

    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
