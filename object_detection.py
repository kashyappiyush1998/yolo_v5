import torch
from PIL import Image
import os
# Model

class YoloV5:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, force_reload=False)

    def get_object_dict_single_image(self, img):
        classes = {}
        imgs = []
        img.filename = 'temp.png'
        imgs.append(img)
        result = self.model(imgs)

        for i, (img, pred) in enumerate(zip(result.imgs, result.pred)):
            if pred is not None:
                for c in pred[:, -1]:
                    n = (pred[:, -1] == c).sum()  # detections per class
                    classes[result.names[int(c)]] = n

        return classes

    def get_object_dict_multiple_images(self, imgs):
        classes = {}
        for img in imgs:
            img.filename = 'temp.png'
        result = self.model(imgs)

        for i, (img, pred) in enumerate(zip(result.imgs, result.pred)):
            if pred is not None:
                for c in pred[:, -1]:
                    n = (pred[:, -1] == c).sum()  # detections per class
                    classes[result.names[int(c)]] = n

        return classes