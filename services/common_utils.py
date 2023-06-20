import numpy as np


def build_data_dict(name, label, bbox, score):
    data = {}
    data['name'] = name
    data['x_center'] = (bbox[0] + bbox[2]) / 2
    data['y_center'] = (bbox[1] + bbox[3]) / 2
    data['width'] = bbox[2] - bbox[0]
    data['height'] = bbox[3] - bbox[1]
    data['label'] = label
    data['score'] = score * 100    
    data['is_occluded'] = 0
    data['is_truncated'] = 0
    data['d3_separation'] = 0
    data['l_label'] = 'None '
    data['r_label'] = 'None '
    data['is_rider_on_2_wheels'] = 0
    return data
