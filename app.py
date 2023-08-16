from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import face_recognition
import cv2
import threading
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 默认摄像头
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
SKIP_FRAMES = 2
socketio = SocketIO(app)

global_frame = None
frame_lock = threading.Lock()

def load_known_faces(directory):
    known_encodings = []
    known_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

def process_frames():
    global global_frame

    # Load known faces from directory
    known_face_encodings, known_face_names = load_known_faces("/Users/liudaijie/Desktop/Live Video Streaming Service with Facial Recognition/")

    frame_count = 0
    while True:
        ret, frame = camera.read()
        if frame_count % SKIP_FRAMES == 0:
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                if True in matches:
                    name = known_face_names[matches.index(True)]
                    # Emit the name of the recognized face using SocketIO
                    socketio.emit('face_detected', {'name': name})

                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            
            with frame_lock:
                global_frame = frame
        frame_count += 1

def generate_frames():
    while True:
        with frame_lock:
            if global_frame is not None:
                ret, buffer = cv2.imencode('.jpg', global_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

t = threading.Thread(target=process_frames)
t.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    try:
        socketio.run(app, debug=True)
    finally:
        camera.release()
