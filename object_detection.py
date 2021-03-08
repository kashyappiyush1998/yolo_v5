import torch
from PIL import Image
import os
from yolov5 import YOLOv5
# Model

class YoloV5:
    def __init__(self):
        # self.model_path = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, force_reload=False)
        self.model_path = '/datadrive/weights/yolov5x.pt'
        self.device = 'cpu'
        self.model = YOLOv5(self.model_path, self.device)

    def get_object_dict_single_image(self, img):
        classes = {}
        imgs = []
        img = Image.fromarray(img)
        img.filename = 'temp.png'
        imgs.append(img)
        result = self.model.predict(imgs, size=1280, augment=True)

        for i, (img, pred) in enumerate(zip(result.imgs, result.pred)):
            if pred is not None:
                for c in pred[:, -1]:
                    n = (pred[:, -1] == c).sum()  # detections per class
                    classes[result.names[int(c)]] = n.data.tolist()

        return classes

    def get_object_dict_multiple_images(self, imgs):
        classes = {}
        new_imgs = []
        for img in imgs:
            img=Image.fromarray(img)
            img.filename = 'temp.png'
            new_imgs.append(img)
        result = self.model.predict(imgs)

        for i, (img, pred) in enumerate(zip(result.imgs, result.pred)):
            if pred is not None:
                for c in pred[:, -1]:
                    n = (pred[:, -1] == c).sum()  # detections per class
                    classes[result.names[int(c)]] = n

        return classes