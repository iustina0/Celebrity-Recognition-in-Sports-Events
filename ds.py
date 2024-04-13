import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((output_channels, 1))
        self.stride = stride
        self.padding = padding

    def forward(self, input):
        batch_size, input_channels, input_height, input_width = input.shape
        output_channels, _, kernel_size, _ = self.weights.shape

        padded_input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        output_height = (input_height - kernel_size + 2*self.padding) // self.stride + 1
        output_width = (input_width - kernel_size + 2*self.padding) // self.stride + 1

        output = np.zeros((batch_size, output_channels, output_height, output_width))

        for b in range(batch_size):
            for c_out in range(output_channels):
                for h in range(0, input_height - kernel_size + 1, self.stride):
                    for w in range(0, input_width - kernel_size + 1, self.stride):
                        output[b, c_out, h // self.stride, w // self.stride] = np.sum(
                            padded_input[b, :, h:h+kernel_size, w:w+kernel_size] * self.weights[c_out, :, :, :]
                        ) + self.bias[c_out]

        return output

# Example usage
input_channels = 3
output_channels = 64
kernel_size = 3
stride = 1
padding = 1

conv_layer = ConvLayer(input_channels, output_channels, kernel_size, stride, padding)



class RegionProposalNetwork:
    def __init__(self, input_channels, anchor_scales, ratios):
        self.anchor_scales = anchor_scales
        self.ratios = ratios
        self.num_anchors = len(anchor_scales) * len(ratios)
        self.conv_layer = ConvLayer(input_channels, 256, 3, 1, 1)  # Example convolutional layer for RPN

    def generate_anchors(self, feature_map_height, feature_map_width, stride):
        anchors = []
        for scale in self.anchor_scales:
            for ratio in self.ratios:
                width = scale * np.sqrt(ratio)
                height = scale / np.sqrt(ratio)
                for i in range(feature_map_height):
                    for j in range(feature_map_width):
                        x_center = j * stride + stride / 2
                        y_center = i * stride + stride / 2
                        anchors.append([x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2])
        return np.array(anchors)

    def forward(self, feature_map):
        batch_size, _, feature_map_height, feature_map_width = feature_map.shape

        anchors = self.generate_anchors(feature_map_height, feature_map_width, stride=1)
        num_anchors = len(anchors)

        # Reshape feature map for the convolutional layer
        reshaped_feature_map = feature_map.reshape((batch_size, -1, feature_map_height, feature_map_width))

        # Apply convolutional layer to the reshaped feature map
        rpn_conv_output = self.conv_layer.forward(reshaped_feature_map)

        # Perform RPN classification and regression here...
        # This is where you would typically have classification and regression branches for each anchor

        # Example output shape, adjust according to your RPN design
        rpn_cls_scores = np.random.rand(batch_size, num_anchors, 2)  # Example classification scores
        rpn_bbox_deltas = np.random.rand(batch_size, num_anchors, 4)  # Example bounding box deltas

        return rpn_cls_scores, rpn_bbox_deltas

# Example usage
input_channels = 512  # Output channels from the last convolutional layer in the backbone network
anchor_scales = [128, 256, 512]
ratios = [0.5, 1, 2]
rpn = RegionProposalNetwork(input_channels, anchor_scales, ratios)
feature_map = np.random.randn(1, input_channels, 32, 32)  # Example feature map from the backbone network
rpn_cls_scores, rpn_bbox_deltas = rpn.forward(feature_map)
print("RPN classification scores shape:", rpn_cls_scores.shape)
print("RPN bounding box deltas shape:", rpn_bbox_deltas.shape)


image_path = 'C:\\Users\\Iustina\Documents\\GitHub\\Celebrity-Recognition-in-Sports-Events\\face_recognition\\data\\face_detection_data\\images\\00000068.jpg'
image = cv2.imread(image_path)

# Preprocess the image (resize, normalize, convert to RGB if necessary)
# Assuming your network takes images of fixed size, you need to resize your image accordingly

# Run the image through your network to get region proposals
# Assuming you have a function to run the RPN on an image and get the bounding box predictions
# Here, I'm using a dummy example
rpn_cls_scores = np.random.rand(1, 500, 2)  # Example classification scores
rpn_bbox_deltas = np.random.rand(1, 500, 4)  # Example bounding box deltas

# Draw bounding boxes on the image based on the region proposals
def draw_boxes(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Convert RPN outputs to bounding boxes
# Here, you would use the predicted bounding box deltas and apply them to the anchor boxes to get the final bounding boxes
# For simplicity, I'm using random bounding box deltas
# You should replace this with the actual logic to apply bounding box deltas to anchors
def convert_to_boxes(anchors, deltas):
    # This is just a dummy conversion, you need to implement the actual logic
    return anchors

# Generate anchor boxes
anchors = rpn.generate_anchors(image.shape[0], image.shape[1], stride=1)

# Convert RPN outputs to bounding boxes
boxes = convert_to_boxes(anchors, rpn_bbox_deltas[0])
print(boxes.shape)
# Get 50 random instances from boxes
random_boxes = random.sample(list(boxes), 50)
# Draw bounding boxes on the image
draw_boxes(image, random_boxes)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()