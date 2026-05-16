import cv2
import matplotlib.pyplot as plt
from pathlib import Path

img_path = Path("D:\\Assessment_VisDrone\\VisDrone_Dataset\\VisDrone2019-DET-train\\images\\0000056_00727_d_0000111.jpg")
label_path = Path("D:\\Assessment_VisDrone\\VisDrone_Dataset\\VisDrone2019-DET-train\\labels\\0000056_00727_d_0000111.txt")

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

with open(label_path, "r") as f:
    for line in f:
        cls, x, y, bw, bh = map(float, line.strip().split())
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, str(int(cls)), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis("off")
plt.show()