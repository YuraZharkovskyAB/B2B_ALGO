import os
import pandas as pd
import numpy as np
from services.common_utils import build_data_dict


VALID_LABELS = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']
AB_LABELS = {'person': 'PEDESTRIAN', 'car': 'CAR', 'bus': 'BUS', 'truck': 'TRUCK', 'bicycle': 'MOTOR', 'motorcycle': 'MOTOR'}

def from_result(result):
    labels, bboxes, scores = [], [], []
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        bbox = box.xyxy[0].tolist()
        score = round(box.conf[0].item(), 2)
        labels.append(label)
        bboxes.append(bbox)
        scores.append(score)
    return np.array(labels), np.array(bboxes), np.array(scores)


def to_detections(records):
    names = [os.path.basename(path) for path, _ in records]
    datas_list = []
    for path, result in records:
        name = os.path.basename(path)
        labels, bboxes, scores = from_result(result)
        keep  = [label in VALID_LABELS for label in labels]
        labels, bboxes, scores = labels[keep], bboxes[keep], scores[keep]
        labels = [AB_LABELS[label] for label in labels]
        datas = [build_data_dict(name, label, bbox, score) for label, bbox, score in zip(labels, bboxes, scores)]
        datas_list.extend(datas)
    return pd.DataFrame.from_dict(datas_list)