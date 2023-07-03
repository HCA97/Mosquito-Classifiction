import os
import csv

import pandas as pd
from ultralytics import YOLO
import torch as th

yolo_model = 'runs/detect/train7/weights/best.pt'
test_dir = '../data/test'

test_df = pd.read_csv('../data/test_phase1_v2.csv')


labels = [
    "albopictus",
    "culex",
    "japonicus/koreicus",
    "culiseta",
    "anopheles",
    "aegypti",
]

rows = []

with th.no_grad():
    model = YOLO(yolo_model)

    for i in range(len(test_df)):

        f_name, img_w, img_h = test_df.iloc[i]
        results = model(os.path.join(test_dir, f_name))

        bbox = [0, 0, img_w, img_h]
        label = 'albopictus'
        conf = 0.0

        for result in results:
            _bbox = [0, 0, img_w, img_h]
            _label = 'albopictus'
            _conf = 0.0

            bboxes_tmp = result.boxes.xyxy.tolist()
            labels_tmp = result.boxes.cls.tolist()
            confs_tmp = result.boxes.conf.tolist()

            for bbox_tmp, label_tmp, conf_tmp in zip(bboxes_tmp, labels_tmp, confs_tmp):
                if conf_tmp > _conf:
                    _bbox = bbox_tmp
                    _label = labels[int(label_tmp)]
                    _conf = conf_tmp

            if _conf > conf:
                bbox = _bbox
                label = _label
                conf = _conf

        rows.append({
            'img_fName': f_name,
            'img_w': img_w, 
            'img_h': img_h,
            'bbx_xtl': int(bbox[0]),
            'bbx_ytl': int(bbox[1]),
            'bbx_xbr': int(bbox[2]),
            'bbx_ybr': int(bbox[3]),
            'class_label': label
        })
        

with open('submissions.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)