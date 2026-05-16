from ultralytics import YOLO
import cv2

model = YOLO("D:\\Assessment_VisDrone\\yolo_results\\train-3\\weights\\best.pt")

img = cv2.imread(r"E:\A&P\DJI\Basketball match\DJI_0046.JPG")

results = model(img)[0]

names = model.names

people_count = 0
pedestrian_count = 0

for box in results.boxes:
    cls = int(box.cls[0])
    label = names[cls]

    if label == "people":
        people_count += 1
    elif label == "pedestrian":
        pedestrian_count += 1

cv2.imwrite("detected_sample.jpg", results)
img = results.plot()

cv2.putText(img, f"People: {people_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)

cv2.putText(img, f"Pedestrians: {pedestrian_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)
cv2.imwrite("detected_sample.jpg", img)
cv2.imshow("Result", img)
cv2.waitKey(0)