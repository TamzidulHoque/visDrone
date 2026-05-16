Dataset:
\-----------

1. The dataset is full of aerial images. Dataset of close objects and COCO datasets are not available.
2. Very unclear details while eveing and night views.
3. Desity of traffic and people are a lot but still detectable and fine.
4. From the kaggle, already .yaml file is given and the conversion files (labels) are available instead of giving annotations.Label contains the parameters of bounding boxes and classes of objects. So a perfect dataset for YOLO applicatoin.Choosing YOLOv8n for industry level performance.
5. 4 duplicates found. 0 corruptions, and rest of the images are perfectly sorted according to label.YOLO can automatically find and remove the duplicates.



Preprocessing:
\--------------

1. YOLOv8 automatically has done some augmentations during the runtime. So now manual augmentation needed.
2. YOLOv8 needed some sortings of locations of the datasets like all the images should be at one folder and the labels are at the other. YOLOv8 downloaded some -dev files and saved the sorted dataset in a destined directory naming as "datasets".
3. .yaml file all the classes mentioned there are being trained because modifying it to only two classes (people, cars) may face overfitting or degrading as the dataset VisDrone is a multi-class dataset by default in its design.


Training
\-------------
1. First running used CPU and was taking more than 6 hrs. So had to install nvidia CUDA as well as newer torch version as CUDA was not installing at the older one.   
2. With GPU RTX 3050, 50 epochs completed in 2.508 hours.
3. 

Detection:
\----------------
1. Clear image/video, very well detection. Struggles to detect while too much far or blurry image.
2. Any undesired object similar to the classes  detects incorrectly.Experiencing the lack of COCO 
datasets and closer object detection image datasets.
3. Pedestrian and people both are considering same class in reality as human. That's why showing both counts in result.
# CV2 in loop:
1. Big challenge for far and smaller objects.
2. Wrong detections found. Model is not very well trained. More epoch or batches may solve this.
# Byte-track and deep-sort and Bot-sort:
1. Their performance are similar. They can distinguish the crowd density 
but still misses a single pedestrian and does faulty detections.