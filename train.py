from ultralytics import YOLO

if __name__ == '__main__':
    print("Loading Base Architecture...")
    model = YOLO("yolov8m.pt")
    
    print("Initiating Training Sequence...")
    model.train(
        data="Railway Crack Detection.yolov8/data.yaml",
        epochs=50,
        imgsz=640,
        patience=10,
    )
    
    print("Validating Final Checkpoint...")
    model.val()
