from ultralytics import YOLO
import cv2

model = YOLO("yolo_results\\train-3\\weights\\best.pt")

img_path = r"samples\input_samples\9999946_00000_d_0000032.jpg"
img = cv2.imread(img_path)

results = model(img)[0]

names = model.names

human_count = 0

for box in results.boxes:
    cls = int(box.cls[0])
    label = names[cls]

    if label == "people" or label == "pedestrian":
        human_count += 1

img = results.plot()

cv2.putText(img,
            f"Human Count: {human_count}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2)

cv2.imwrite("detected_sample.jpg", img)

scale = 0.6
img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()