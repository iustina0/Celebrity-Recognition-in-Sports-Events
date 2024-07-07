import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from FaceDetection import face_detect
from FaceRecognition import face_recogniton
from tqdm import tqdm
import sys
import os
import time
import socketio
from skimage.metrics import structural_similarity as ssim


sio = socketio.Client()

@sio.event
def connect():
    print('Connection established')

@sio.event
def disconnect():
    print('Disconnected from server')


class Face:
    def __init__(self, image, bbox, identity=None):
        self.image = image
        self.bbox = bbox
        self.identity = identity

def compare_faces(face1, face2):
    face1_embedding = face_recogniton.get_embedding(Image.fromarray(face1).convert('RGB'))
    face1_embedding = torch.tensor(face1_embedding).unsqueeze(0)
    face2_embedding = face_recogniton.get_embedding(Image.fromarray(face2).convert('RGB'))
    face2_embedding = torch.tensor(face2_embedding).unsqueeze(0)
    distance = F.pairwise_distance(face1_embedding, face2_embedding)
    return distance.item()

def crop_to_aspect_ratio(image, bbox, target_ratio=0.725):
    image_height, image_width = image.shape[:2]
    
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    width = x4 - x1
    height = y3 - y4

    new_width = int(height * target_ratio)

    if new_width < width:
        delta = (width - new_width) // 2
        buffer_height = height // 20
        buffer_width = new_width // 20
        
        y1 = max(0, y1 - buffer_height)
        y2 = min(image_height, y2 + buffer_height)
        y3 = min(image_height, y3 + buffer_height)
        y4 = max(0, y4 - buffer_height)

        x1 = max(0, x1 + delta - buffer_width)
        x2 = max(0, x2 + delta - buffer_width)
        x3 = min(image_width, x3 - delta + buffer_width)
        x4 = min(image_width, x4 - delta + buffer_width)

    min_x = max(0, int(min(x1, x2, x3, x4)))
    max_x = min(image_width, int(max(x1, x2, x3, x4)))
    min_y = max(0, int(min(y1, y2, y3, y4)))
    max_y = min(image_height, int(max(y1, y2, y3, y4)))

    cropped_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
    return cropped_box


def detect_scene_change_histogram(frame1, frame2, threshold=0.7):
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    hist_frame1 = cv2.calcHist([hsv_frame1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_frame2 = cv2.calcHist([hsv_frame2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    cv2.normalize(hist_frame1, hist_frame1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_frame2, hist_frame2, 0, 1, cv2.NORM_MINMAX)
    
    similarity = cv2.compareHist(hist_frame1, hist_frame2, cv2.HISTCMP_CORREL)
    
    scene_change = similarity < threshold 
    
    return scene_change, similarity

def process_video(input_path, output_path, threshold=0.7):
    try:
        sio.connect('http://localhost:5000', wait_timeout=10) 

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file: {input_path}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        prev_faces = []
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_detect.detect_face(processed_frame)
            boxes = np.array(boxes, dtype=int)
            cropped_boxes = []
            for box in boxes:
                cropped_box = crop_to_aspect_ratio(processed_frame, box)
                cropped_boxes.append(cropped_box)
            faces = [Face(processed_frame[box[0][1]:box[2][1], box[0][0]:box[2][0]], box) for box in cropped_boxes]

            current_faces = []

            if prev_frame is not None:
                scene_change, sim = detect_scene_change_histogram(prev_frame, frame)
                prev_frame = frame
                if scene_change:
                    print(f"Scene change detected with SSIM: {sim}")
                    for face in faces:
                        identity, _ = face_recogniton.recognize_faces([Image.fromarray(face.image).convert('RGB')])
                        face.identity = identity[0]
                        current_faces.append(face)

                        cv2.rectangle(frame, face.bbox[0], face.bbox[2], (0, 0, 255), 2)
                        cv2.putText(frame, identity[0], (face.bbox[0][0] - 10, face.bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    for face in faces:
                        identity = None
                        min_distance = float('inf')

                        for prev_face in prev_faces:
                            distance = compare_faces(face.image, prev_face.image)
                            if distance < threshold and distance < min_distance:
                                min_distance = distance
                                identity = prev_face.identity

                        if identity is None:
                            identity, _ = face_recogniton.recognize_faces([Image.fromarray(face.image).convert('RGB')])
                            identity = identity[0]

                        face.identity = identity
                        current_faces.append(face)

                        cv2.rectangle(frame, face.bbox[0], face.bbox[2], (0, 0, 255), 2)
                        cv2.putText(frame, identity, (face.bbox[0][0] - 10, face.bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                for face in faces:
                    identity, _ = face_recogniton.recognize_faces([Image.fromarray(face.image).convert('RGB')])
                    face.identity = identity[0]
                    current_faces.append(face)

                    cv2.rectangle(frame, face.bbox[0], face.bbox[2], (0, 0, 255), 2)
                    cv2.putText(frame, identity[0], (face.bbox[0][0] - 10, face.bbox[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    prev_frame = frame
            
            out.write(processed_frame)
            processed_frames += 1
            progress = (processed_frames / total_frames) * 100
            print(f"Progress: {progress:.2f}%")
            sio.emit('progress', {'progress': progress})


        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if os.path.exists(output_path):
            print(f"Processed video saved: {output_path}")
        else:
            print("Error: Processed video file was not created")

        sio.emit('processing_complete', {'filename': os.path.basename(output_path)})

    except socketio.exceptions.ConnectionError as e:
        print(f"ConnectionError: {e}")

    finally:
        sio.disconnect()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_video(input_path, output_path)
