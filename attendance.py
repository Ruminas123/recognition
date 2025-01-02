import cv2
import numpy as np
import face_recognition
import os
import threading
import queue
from datetime import datetime
from pymongo import MongoClient
import gridfs
import tkinter as tk
from PIL import Image, ImageTk
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB connection string
DATABASE_NAME = "face_recognition"
COLLECTION_NAME = "DetectionLogs"

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
log_collection = db[COLLECTION_NAME]
fs = gridfs.GridFS(db)
frame_queue = queue.Queue(maxsize=10)

class FaceRecognitionSystem:
    def __init__(self, rtsp_url):
        self.directory = "ImagesAttendance"
        os.makedirs(self.directory, exist_ok=True)
        self.cap = None
        self.rtsp_url = rtsp_url
        self.is_running = False
        self.frame = None
        self.encodeListKnown = []
        self.classNames = []
        self.images = []
        self.blacklist_image = False

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)
            if encode:  # Check if faces are detected before appending
                encodeList.append(encode[0])
        return encodeList

    def initializeKnownFaces(self):
        """
        Load and encode known faces from the directory.
        """
        self.myList = os.listdir(self.directory)
        print(f"Images found in directory: {self.myList}")
        valid_images = []
        self.classNames = []

        for filename in self.myList:
            file_path = os.path.join(self.directory, filename)
            image = cv2.imread(file_path)

            if image is not None:
                valid_images.append(image)
                self.classNames.append(os.path.splitext(filename)[0])
            else:
                print(f"Warning: Could not read file {file_path}. Skipping...")

        if not valid_images:
            print("No valid images found in the directory!")
            return

        self.encodeListKnown = FaceRecognitionSystem.findEncodings(valid_images)
        print(f"Initialized {len(self.encodeListKnown)} known faces from {len(valid_images)} valid images.")


    def capture_frames(self, cap):
        while True:
            ret, frame = cap.read()
            if ret:
                if frame_queue.full():
                    frame_queue.get()  # Remove the oldest frame
                frame_queue.put(frame)
                del frame  # Free memory
            else:
                print("Failed to capture frame.")
                break

    def start_recognition(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.is_running = True

        capture_thread = threading.Thread(target=self.capture_frames, args=(self.cap,))
        process_thread = threading.Thread(target=self.process_frames)
        
        capture_thread.daemon = True
        process_thread.daemon = True
        
        capture_thread.start()
        process_thread.start()

    def markAttendance(name):
        with open('attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [entry.split(',')[0] for entry in myDataList]
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

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

    def process_frames(self):
        frame_count = 0 
        while True:
            if not frame_queue.empty():
                frame_count += 1  # Increment frame count
                if frame_count % 2 != 0:  # Skip processing if not every 2nd frame
                    continue

                if self.blacklist_image:
                    self.initializeKnownFaces()
                    self.blacklist_image = False

                img = frame_queue.get()
                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for speed
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB
                faceCurFrame = face_recognition.face_locations(imgS)
                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
                
                def save_image_face(file_name, face_image):
                    new_file_path = os.path.join(self.directory, file_name)
                    face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)  # Convert to BGR
                    cv2.imwrite(new_file_path, face_image_bgr)
                    self.initializeKnownFaces()

                for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                    if not self.encodeListKnown:  # No known faces
                        print("No known faces found. Adding first face...")
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up
                        face_image = img[y1:y2, x1:x2]
                        save_image_face("1.jpg", face_image)
                        continue

                    faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                    if len(faceDis) == 0:  # Ensure faceDis is not empty
                        print("No face distances available. Skipping...")
                        continue

                    matchIndex = np.argmin(faceDis)
                    face_percent_value = 1 - faceDis[matchIndex]
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale up

                    if face_percent_value < 0.45:
                        face_image = img[y1:y2, x1:x2]  # Crop the face region
                        new_file_name = f"{len(self.encodeListKnown) + 1}.jpg"
                        save_image_face(new_file_name, face_image)

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)

                    if face_percent_value >= 0.5:
                        name = self.classNames[matchIndex].upper()
                        cv2.putText(img, f"{name} : {face_percent_value:.2}%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                        FaceRecognitionSystem.markAttendance(name)
                    else:
                        name = "Unknown"
                        cv2.putText(img,  f"{name} : {face_percent_value:.2}%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

                    timestamp = datetime.now()
                    FaceRecognitionSystem.log_detection_to_mongo(name, face_percent_value, timestamp)
                self.frame = img
                

    def open_file(self):
        """Open file dialog to select an image and save it."""
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            # Read the selected image
            image = cv2.imread(file_path)
            if image is not None:
                # Save it to ImagesAttendance
                filename = os.path.basename(file_path)  # Extract the file name
                self.save_image(image, filename)

    def save_image(self, image, filename):
        try:
            self.blacklist_image = True
            print("blacklist_image",self.blacklist_image)
            new_file_path = os.path.join(self.directory, filename)
            cv2.imwrite(new_file_path, image)
            print(f"Image saved to {new_file_path}")
            # self.initializeKnownFaces()
        except Exception as e:
            print(f"Failed to save image: {e}")

def generate_frames(rtsp_url):
    face_system = FaceRecognitionSystem(rtsp_url)
    print("Initializing known faces...")
    face_system.initializeKnownFaces()
    print("Starting recognition...")

    root = tk.Tk()
    label = tk.Label(root)
    open_file_button = tk.Button(root, text="Open Image File", command=face_system.open_file)
    open_file_button.pack(pady=20)

    label = tk.Label(root)
    label.pack()

    try:
        face_system.start_recognition()
        while face_system.is_running:
            if face_system.frame is not None:
                frame = face_system.frame
                # Resize frame to 450px width, maintaining aspect ratio
                height, width = frame.shape[:2]
                new_width = 900
                new_height = int((new_width / width) * height)
                resized_frame = cv2.resize(frame, (new_width, new_height))

                # Convert the resized frame from BGR to RGB
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

                # Convert to Image and update Tkinter label
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)
                
                # Update the label with the new image
                label.img_tk = img_tk
                label.config(image=img_tk)

            root.update_idletasks()
            root.update()

    except Exception as e:
        print(f"Error during face recognition: {e}")
    finally:
        print("End face recognition...")

    root.mainloop()

if __name__ == "__main__":
    # rtsp_url = "rtsp://stream5:Abc12345@conic.myds.me:8005/Streaming/channels/501"
    rtsp_url = "rtsp://admin:Abc12345@192.168.100.14/streaming/channels/101"
    generate_frames(rtsp_url)
