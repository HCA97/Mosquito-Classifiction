from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/home/cemmi/Documents/aicrowd-mos/data_yolo_cls', epochs=3, imgsz=128, dropout=0.5)