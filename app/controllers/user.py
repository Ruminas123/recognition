from flask import request, jsonify
from ..models.user import create_user, get_user

def insert_user(data):
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    # Check if the user already exists in MongoDB
    if get_user(username):
        return jsonify({"error": "User already exists."}), 400

    # Create a new user in MongoDB
    create_user({'username': username,'password': password})
    return jsonify({"message": f"User {username} created successfully!"}), 201