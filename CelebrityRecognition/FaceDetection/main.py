from FaceDetection.model.fasterRCNN import FasterRCNN
from FaceDetection.model.trainer import FasterRCNNTrainer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model = FasterRCNN()
trainer = FasterRCNNTrainer(model).cuda()
trainer.load("CelebrityRecognition/FaceDetection/checkpoints/fasterrcnn_pretrained-06121613.pth")

FasterRCNNModel = trainer.faster_rcnn
FasterRCNNModel.eval()


