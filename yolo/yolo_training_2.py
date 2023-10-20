from ultralytics import YOLO

# Load a model
model = YOLO(
    "runs/detect/train11/weights/last.pt"
)  # load a pretrained model (recommended for training)

model.train(
    data="./yolo_config_mos.yml",
    epochs=200,
    imgsz=640,
    shear=0.5,
    degrees=15,
    perspective=0.001,
    mosaic=1,
    translate=0.2,
    flipud=0.3,
    close_mosaic=120,
    resume=True,
)
