import random
import numpy as np
import cv2


# this is the model we'll be using for 
# object detection
# from keras.applications import Xception
 
# # for preprocessing the input
# from keras.applications.xception import preprocess_input
# from keras.applications import imagenet_utils
# from keras.preprocessing.image import img_to_array
# read the input image
img = cv2.imread('C:\\Users\\Iustina\Documents\\GitHub\\Celebrity-Recognition-in-Sports-Events\\face_recognition\\data\\face_detection_data\\images\\00000085.jpg')
H = img.shape[0]
W = img.shape[1]
# instantiate the selective search
# segmentation algorithm of opencv
search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
def calculate_iou(box1, box2):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(box1Area + box2Area - interArea)

    # Return the intersection over union value
    return iou
# set the base image as the input image
search.setBaseImage(img)

# since we'll use the fast method we set it as such
search.switchToSelectiveSearchFast()

# you can also use this for more accuracy:
# search.switchToSelectiveSearchQuality()
rects = search.process() # process the image

random_boxes = random.sample(list(rects), 50)
roi = img.copy()
for (x, y, w, h) in random_boxes:

	# Check if the width and height of
	# the ROI is atleast 10 percent
	# of the image dimensions and only then
	# show it
	if (w / float(W) < 0.1 or h / float(H) < 0.1):
		continue

	# Let's visualize all these ROIs
	cv2.rectangle(roi, (x, y), (x + w, y + h),
				(0, 200, 0), 2)

roi = cv2.resize(roi, (640, 640))
final = cv2.hconcat([cv2.resize(img, (640, 640)), roi])
cv2.imshow('ROI', final)
cv2.waitKey(0)
