from typing import Optional, Dict

import numpy as np
import cv2
import torch


class FeatureDetector:

    def __init__(self, item_dict: Dict[str, np.ndarray]):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher()
        self.class_list, self.desc_list = self.extract_descriptors(item_dict)

        self.MATCHES_THRESH = 80

    @staticmethod
    def extract_descriptors(self, item_dict):
        class_list = []
        desc_list = []
        for key, val in item_dict:
            _, queryDescriptors = self.orb.detectAndCompute(item_dict, None)
            class_list.append(key)
            desc_list.append(queryDescriptors)

        return class_list, desc_list

    def identify_object(self, obj_img):
        _, queryDescriptors = self.orb.detectAndCompute(obj_img, None)

        matches = [self.matcher(queryDescriptors, cls_desc) for cls_desc in self.desc_list]
        matches_num = [len(match) for match in matches]

        highest_idx = matches_num.index(max(matches_num))

        return self.class_list[highest_idx] if highest_idx > self.MATCHES_THRESH else None


if __name__ == "__main__":

    yolo = torch.hub.load("ultralytics/yolov5", "custom", path="./best.pt")

    cap = cv2.VideoCapture(0)

    ret = True
    while ret:
        # capture frame-by-frame
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_list = yolo(frame_rgb).xyxy[0].cpu()

        for pred in pred_list:
            bbox = list(int(np.round(x)) for x in pred[:4])
            score = pred[4]
            class_id = pred[5]

            if class_id == 0:
                outline = (255, 165, 0)
            elif class_id == 1:
                outline = (0, 100, 100)

            print(bbox)
            cv2.rectangle(frame, bbox[:2], bbox[2:], outline, 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





