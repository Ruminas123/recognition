# app/routes.py
import os
from flask import Blueprint, Response, request, jsonify, render_template, redirect, url_for
from ..models.user import create_user, get_user
from ..models.face_recognition import generate_frames
from ..controllers.user import insert_user
from werkzeug.utils import secure_filename

main_routes = Blueprint('main_routes', __name__)

UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../uploads'))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@main_routes.route('/')
def home():
    return render_template('index.html')

@main_routes.route('/create_user', methods=['POST'])
def create_user_route():
    return insert_user(request.json)

@main_routes.route('/video_feed')
def video_feed():
    # rtsp = 'rtsp://stream5:Abc12345@conic.myds.me:8005/Streaming/channels/501'
    rtsp = "rtsp://admin:Abc12345@192.168.100.14/streaming/channels/101"
    return Response(generate_frames(rtsp), mimetype='multipart/x-mixed-replace; boundary=frame')

@main_routes.route('/upload', methods=['POST'])
def uploader():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    try:
        # Secure and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Redirect to home page (index.html) after successful upload
        return redirect(url_for('main_routes.home'))  # Redirect to the home route (index.html)
    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500