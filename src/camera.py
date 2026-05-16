from ultralytics import YOLO

model = YOLO(r"D:\Assessment_VisDrone\yolo_results\train-3\weights\best.pt")

model.track(
    source=0,
    show=True,
    save=True,
    persist=True,
    tracker="botsort.yaml"
)