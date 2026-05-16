from ultralytics import YOLO
import cv2
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Deep-sort
# video_path = r"E:\A&P\DJI\Basketball match\DJI_0044.MP4"
# output_path = r"tracked_output.mp4"
# model_path = r"D:\Assessment_VisDrone\yolo_results\train-3\weights\best.pt"

# model = YOLO(model_path)
# tracker = DeepSort(max_age=30)

# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# writer = cv2.VideoWriter(
#     output_path,
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     (width, height)
# )

# names = model.names

# while cap.isOpened():
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     detections = []

#     for box in results.boxes:
#         conf = float(box.conf.item())

#         if conf < 0.25:
#             continue

#         x1, y1, x2, y2 = map(int, box.xyxy[0])

#         cls_id = int(box.cls.item())
#         class_name = names[cls_id]

#         detections.append(
#             ([x1, y1, x2 - x1, y2 - y1], conf, class_name)
#         )

#     tracks = tracker.update_tracks(detections, frame=frame)

#     for track in tracks:

#         if not track.is_confirmed():
#             continue

#         track_id = track.track_id
#         class_name = track.det_class

#         ltrb = track.to_ltrb()
#         x1, y1, x2, y2 = map(int, ltrb)

#         label = f"{class_name} | ID {track_id}"

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         cv2.putText(
#             frame,
#             label,
#             (x1, max(0, y1 - 10)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 0),
#             2
#         )

#     writer.write(frame)

#     cv2.imshow("Tracked Video", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# writer.release()
# cv2.destroyAllWindows()
#===================================================================================================
#bytrack and botsort
model = YOLO(r"D:\Assessment_VisDrone\yolo_results\train-3\weights\best.pt")
model.track(
    source="E:\A&P\DJI\Basketball match\DJI_0044.MP4",
    show=True,
    tracker="bytetrack.yaml",
    # tracker="botsort.yaml",
    persist=True,
    save=True,
    project="D:/Assessment_VisDrone/yolo_results",
    name="tracked_video"
)
#==================================================================================================
# cv2 in loop
# video_path = r"E:\A&P\DJI\Basketball match\DJI_0044.MP4"
# output_path = r"D:\Assessment_VisDrone\output_video.mp4"
# model_path = r"D:\Assessment_VisDrone\yolo_results\train-3\weights\best.pt"
# model = YOLO(model_path)
# cap = cv2.VideoCapture(video_path)

# fps = cap.get(cv2.CAP_PROP_FPS)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# names = model.names

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)[0]

#     people_count = 0
#     pedestrian_count = 0

#     for box in results.boxes:
#         cls = int(box.cls.item())
#         label = names[cls]

#         if label == "people":
#             people_count += 1
#         elif label == "pedestrian":
#             pedestrian_count += 1

#     annotated = results.plot()

#     cv2.putText(annotated, f"People: {people_count}",
#                 (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 0, 255), 2)

#     cv2.putText(annotated, f"Pedestrians: {pedestrian_count}",
#                 (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
#                 1, (0, 0, 255), 2)

#     writer.write(annotated)
#     cv2.imshow("Result", annotated)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# writer.release()
# cv2.destroyAllWindows()