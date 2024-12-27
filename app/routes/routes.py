# app/routes.py
from flask import Blueprint, request, jsonify, render_template, Response
from ..models.user import create_user, get_user
from ..models.face_recognition import generate_frames
from ..controllers.user import insert_user

main_routes = Blueprint('main_routes', __name__)

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