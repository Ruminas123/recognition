# app/extensions.py

from pymongo import MongoClient
from flask import current_app

client = None  # This will hold the MongoDB client
db = None  # This will hold the database

def init_mongo(app):
    global client, db
    mongo_uri = app.config['MONGO_URI']  # Get MONGO_URI from Flask app config
    client = MongoClient(mongo_uri)  # Create a new MongoDB client
    db = client.face_recognition  # Access the 'face_recognition' database
