from typing import Dict

import numpy as np
import cv2
import torch
from tracker import CentroidTracker


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


class ZoneCounter:

    def __init__(self, zone_dim):
        self.zone_dim = zone_dim
        self.zone_bbox = None
        self.OVERLAP_THRESH = 0.25

        self.in_zone_ids = set()
        self.in_zone_count = 0

    def set_zone_shape(self, frame) -> None:
        h, w, _ = frame.shape
        x1, y1, x2, y2 = self.zone_dim
        self.zone_bbox = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

    def update(self, preds) -> int:
        assert self.zone_bbox is not None, "Zone bounding box not initialized, use method set_zone_shape"

        track_ids = preds[:, 6]
        bboxes = preds[:, :4]

        overlaps = self.calculate_overlap_percentage(self.zone_bbox, bboxes)

        for track_id, overlap in zip(track_ids, overlaps):
            if overlap >= self.OVERLAP_THRESH and track_id not in self.in_zone_ids:
                self.in_zone_ids.add(track_id)

            if overlap <= self.OVERLAP_THRESH and track_id in self.in_zone_ids:
                self.in_zone_ids.remove(track_id)7

        return self.count

    def draw_image(self, frame: np.ndarray) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (30, 30)
        fontScale = 1
        color = (255, 127, 0)
        thickness = 1

        frame = cv2.putText(frame, f"Items in box: {self.count}",
                            org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)

        thickness = 2
        frame = cv2.rectangle(frame, self.zone_bbox[:2], self.zone_bbox[2:], color, 2)
        return frame

    @property
    def count(self):
        return len(self.in_zone_ids)

    @staticmethod
    def calculate_overlap_percentage(ref_box, boxes_list):
        """
        Calculates the percentage of area overlapped between a reference box and a list of bounding boxes.

        Arguments:
        ref_box (tuple): Tuple containing the coordinates of the reference box in the format (x_min, y_min, x_max, y_max).
        boxes_list (list): List of bounding boxes, each represented as a tuple in the same format as ref_box.

        Returns:
        float: The percentage of area overlapped between the reference box and each box in the list.
        """
        ref_x_min, ref_y_min, ref_x_max, ref_y_max = ref_box
        ref_area = (ref_x_max - ref_x_min) * (ref_y_max - ref_y_min)

        overlap_percentages = []
        for box in boxes_list:
            x_min, y_min, x_max, y_max = box

            # Calculate the overlapping area
            overlap_x_min = max(ref_x_min, x_min)
            overlap_y_min = max(ref_y_min, y_min)
            overlap_x_max = min(ref_x_max, x_max)
            overlap_y_max = min(ref_y_max, y_max)

            overlap_area = max(0, overlap_x_max - overlap_x_min) * max(0, overlap_y_max - overlap_y_min)
            overlap_percentage = (overlap_area / ref_area) * 100

            overlap_percentages.append(overlap_percentage)

        return overlap_percentages


if __name__ == "__main__":

    zone_dim = (0.5, 0.5, 1.0, 1.0)
    zone_counter = ZoneCounter(zone_dim)
    tracker = CentroidTracker()
    yolo = torch.hub.load("ultralytics/yolov5", "custom", path="./best.pt")
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    assert ret is True, "No source from webcam"
    zone_counter.set_zone_shape(frame)

    ret = True
    while ret:
        # capture frame-by-frame
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pred_list = yolo(frame_rgb).xyxy[0].cpu()

        obj_bboxes = pred_list[pred_list[:, 5] == 0.0]

        _ , origin_rects = tracker.update(obj_bboxes[:, :4])

        tracked_objs = np.ndarray(shape=(len(origin_rects), 7))
        for i, (obj_id, bbox) in enumerate(origin_rects.items()):
            tracked_objs[i] = [*bbox, 1, 0, obj_id]

        zone_counter.update(tracked_objs)
        frame = zone_counter.draw_image(frame)

        for pred in pred_list:
            bbox = list(int(np.round(x)) for x in pred[:4])
            score = pred[4]
            class_id = pred[5]

            if class_id == 0:
                outline = (255, 165, 0)
            elif class_id == 1:
                outline = (0, 100, 100)

            cv2.rectangle(frame, bbox[:2], bbox[2:], outline, 2)

            obj_bboxes = pred

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
