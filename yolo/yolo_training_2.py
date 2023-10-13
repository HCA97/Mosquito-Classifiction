from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

model.train(
    data="./yolo_config_mos_two_class.yml",
    epochs=200,
    imgsz=640,
    optimizer="AdamW",
    cos_lr=True,
    lr0=1e-3,
    warmup_bias_lr=1e-3,
    warmup_epochs=3,
    shear=0.5,
    degrees=15,
    perspective=0.001,
    mosaic=1,
    translate=0.2,
    flipud=0.3,
    mixup=0.05,
    close_mosaic=120,
    agnostic_nms=True,
)
