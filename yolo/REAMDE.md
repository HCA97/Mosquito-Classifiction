
## sparse ml training

Ref: https://sparsezoo.neuralmagic.com/models/yolov8-m-voc_coco-pruned75?hardware=deepsparse-c6i.12xlarge&tab=3

```bash
sparseml.ultralytics.train \
  --model "zoo:cv/detection/yolov8-m/pytorch/ultralytics/voc/pruned75-none" \
  --recipe "zoo:cv/detection/yolov8-m/pytorch/ultralytics/voc/pruned75-none" \
  --data yolo_config_mos.yml \
  --batch 64 \
  --patience 0
```