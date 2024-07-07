import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from FaceDetection import face_detect
from FaceRecognition import face_recogniton
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim


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


def detect_scene_change_ssim(frame1, frame2, threshold=0.8):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    sim, _ = ssim(gray1, gray2, full=True)
    
    scene_change = sim < threshold
    
    return scene_change, sim

def process_video(input_video_path, output_video_path, threshold=0.7):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

    prev_faces = []
    prev_frame = None

    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            boxes = face_detect.detect_face(frame)
            boxes = np.array(boxes, dtype=int)
            cropped_boxes = []
            for box in boxes:
                cropped_box = crop_to_aspect_ratio(frame, box)
                cropped_boxes.append(cropped_box)
            faces = [Face(frame[box[0][1]:box[2][1], box[0][0]:box[2][0]], box) for box in cropped_boxes]

            current_faces = []

            if prev_frame is not None:
                scene_change, sim = detect_scene_change_ssim(prev_frame, frame)
                if not scene_change:
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


            prev_faces = current_faces
            out.write(frame)
            pbar.update(1)

    cap.release()
    out.release()


if __name__ == '__main__':
    input_video_path = 'tribune.mp4'
    output_video_path = 'output_video.avi'

    process_video(input_video_path, output_video_path)