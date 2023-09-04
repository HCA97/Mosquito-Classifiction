from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="./yolo_config_mos_balance.yml", epochs=100, imgsz=640)

model.train(data="./yolo_config_mos.yml", epochs=100, imgsz=640)
