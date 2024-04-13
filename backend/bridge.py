import sys
import cv2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('processed_video.avi', fourcc, 25.0, (1920 ,1080))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame, 1)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = sys.argv[1]
    process_video(video_path)

