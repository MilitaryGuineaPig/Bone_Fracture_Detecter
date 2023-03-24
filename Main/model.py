from ultralytics import YOLO
import cv2
import cvzone
import sys
import math

model = YOLO("/Users/monkey/Public/Python/Diploma_1/Yolo_models/first.pt")
classNames = ['angle', 'fracture', 'line', 'messed_up_angle']
# read an image
img = cv2.imread(sys.argv[1])
results = model(img, stream=True)

# creating cof for resized img
cof_x = 400/img.shape[1]
cof_y = 400/img.shape[0]
new_size = (400, 400)
resized_image = cv2.resize(img, new_size)


for r in results:
    boxes = r.boxes
    for box in boxes:
        # Bounding Box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1*cof_x-2), int(y1*cof_y-2), int(x2*cof_x+2), int(y2*cof_y+2)
        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h))
        # Confidence
        conf = math.ceil((box.conf[0] * 100)) / 100
        # Class Name
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        print(currentClass)

        if conf > 0.5:
            myColor = (0, 0, 255)  # Red
            cvzone.putTextRect(resized_image, f'{classNames[cls]} {conf}',
                               (max(0, x1), max(35, y1)), scale=1, thickness=2, colorT=(255, 255, 255), offset=1)
            cv2.rectangle(resized_image, (x1, y1), (x2, y2), myColor, 1)


cv2.imwrite("/Users/monkey/Public/Python/Diploma_1/Main/result.jpg", resized_image)

#cv2.imshow("Image", img)
#cv2.waitKey(0)
