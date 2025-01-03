from flask import Flask, render_template, Response, send_from_directory, send_file
import os
from attendance import initializeKnownFaces

app = Flask(__name__)



# Directory where images are stored
IMAGE_DIR = "ImagesAttendance"
MAIN = initializeKnownFaces()

@app.route('/')
def index():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    # Get all image filenames from the directory
    images = [img for img in os.listdir(IMAGE_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return render_template('index.html', images=images)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images/<filename>')
def get_image(filename):
    file_path = os.path.join(IMAGE_DIR, filename)
    return send_file(file_path)

# Route สำหรับสตรีมวิดีโอ
@app.route('/video_feed')
def video_feed():
    return Response(initializeKnownFaces(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)