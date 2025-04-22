

from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import torch
from datetime import datetime

# Base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define folder paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_DIR, 'processed')
ELEPHANT_FRAMES_FOLDER = os.path.join(BASE_DIR, 'elephant_frames')

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(ELEPHANT_FRAMES_FOLDER, exist_ok=True)

# Initialize the Flask app
app = Flask(__name__, template_folder='templates')

# Load the trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/intozi/Desktop/elephant_dataset/yolov5/runs/train/elephant_detector/weights/best.pt')

# Route to serve index page
@app.route('/')
def index():
    return render_template('index.html')

# Handle video upload and detection
@app.route('/start-detection', methods=['POST'])
def start_detection():
    video_file = request.files['video']
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)

    detection_count = 0
    output_filename = f"output_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
    output_video_path = os.path.join(PROCESSED_FOLDER, output_filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Error opening video file."}), 400

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (640, 480))

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]
        elephant_detections = detections[detections['name'] == 'elephant']
        detection_count += len(elephant_detections)

        # Save elephant frames
        if not elephant_detections.empty:
            frame_filename = os.path.join(ELEPHANT_FRAMES_FOLDER, f"elephant_frame_{frame_index}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_index += 1

        # Annotate and resize frame
        frame = results.render()[0]
        frame_resized = cv2.resize(frame, (640, 480))
        out.write(frame_resized)

    cap.release()
    out.release()

    return jsonify({
        'videoUrl': f'/processed/{output_filename}',
        'detectionCount': detection_count
    })

# Route to serve processed video
@app.route('/processed/<filename>')
def serve_processed_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)




