import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
from tracker import Tracker

model = YOLO('best.pt')


cap = cv2.VideoCapture('glass-jars.mov')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
tracker=Tracker()

count = 0
cy1=305
offset=6
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
list3=[]
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

    list=[]
    
    list1=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        
    

        if cy1<(cy+offset) and cy1>(cy-offset):
          
           w=x4-x3
           h=y4-y3
           if 81 <= w <= 100:
              cv2.circle(frame,(cx,cy),4,(255,0,0),-1) 
              cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
              cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
              if list3.count(id)==0:
                 list3.append(id)
    counter=len(list3)
    cv2.line(frame,(463,305),(742,305),(0,0,255),2)
    cvzone.putTextRect(frame, f'Counter:-{counter}', (50, 60), 1, 1)
    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()