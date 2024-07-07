from gevent import monkey
monkey.patch_all()

import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = ''
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return 'CORS enabled for SocketIO with gevent!'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']

    if file.filename == '':
        return 'No selected file', 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        process = subprocess.Popen(['python', 'CelebrityRecognition/process_uploaded_video.py', upload_path, output_path])
        
        return jsonify({'message': 'File uploaded successfully', 'filename': output_filename}), 200

    return 'Unsupported file format', 400

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('processing_complete')
def handle_processing_complete(data):
    filename = data.get('filename')
    if filename:
        print(f"Processing complete for: {filename}")
        socketio.emit('processing_completed', {'filename': filename})


@socketio.on('progress')
def handle_progress(data):
    progress = data.get('progress')
    if progress:
        print(f"Progress: {progress:.2f}%")
        socketio.emit('progress', {'progress': progress})


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(file_path):
        print('Sending file...')
        return send_file(file_path, as_attachment=True)
    else:
        return 'File not found', 404

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO Error: {e}")
    traceback.print_exc()

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled Exception: {str(e)}")
    traceback.print_exc()
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
