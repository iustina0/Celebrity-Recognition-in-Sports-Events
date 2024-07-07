from FaceDetection.model.fasterRCNN import FasterRCNN
from FaceDetection.model.trainer import FasterRCNNTrainer
import numpy as np
import cv2


model = FasterRCNN()
trainer = FasterRCNNTrainer(model).cuda()
trainer.load(".\\FaceDetection\\checkpoints\\fasterrcnn_pretrained-06142009.pth")
faster_rcnn = trainer.faster_rcnn


def draw_red_triangle(img, box):
    pts = np.array([[box[1], box[0]], [box[1], box[2]], [box[3], box[2]], [box[3], box[0]]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)


cap = cv2.VideoCapture(".\\FaceDetection\\video1.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('.\\FaceDetection\\processed_video.avi', fourcc, 25.0, (1280, 720))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame_RBG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_RBG = np.asarray(frame_RBG, dtype=np.float32)
        frame_RBG = frame_RBG.transpose((2, 0, 1))

        _bboxes, _ , _scores = faster_rcnn.predict([frame_RBG], visualize=True)
        # print(_scores[0])

        for i in range(len(_bboxes[0])):
            bboxe = _bboxes[0][i].astype(int)
            score = _scores[0][i]
            if score > 0.9:
                draw_red_triangle(frame, bboxe)
        out.write(frame)
    else:
        break

cap.release()
out.release()
