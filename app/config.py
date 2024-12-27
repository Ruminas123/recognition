from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # SECRET_KEY = os.environ.get('SECRET_KEY', 'your_default_secret_key')
    # MONGO_URI = os.getenv('MONGO_URI', 'mongodb://mongo:27023/face_recognition')
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/face_recognition')

