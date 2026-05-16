from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data="visdrone.yaml",
        epochs= 50,
        imgsz=640,
        batch=8,
        device= 0, 
        project= r"D:\Assessment_VisDrone\yolo_results",
    )