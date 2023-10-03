from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(data="./yolo_config_mos_two_class.yml", epochs=100, imgsz=640)


# model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)

# model.train(data="./yolo_config_mos_mosqutio.yml", epochs=1, imgsz=640)
