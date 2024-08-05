import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone


model = YOLO('best.pt')


cap = cv2.VideoCapture('')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")


count = 0


while True:
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
    frame=cv2.resize(frame,(1020,600))
    

    

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
       
    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()