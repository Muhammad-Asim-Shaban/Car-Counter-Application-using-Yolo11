from ultralytics import YOLO
import cv2
import math
import cvzone
from sort import *

#adding all the class names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
#importing the mask
defaultMask=cv2.imread("mask.png")

#tracking
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
totalCount=[]
limits=[400,297,673,297]




# for stored videos
cap=cv2.VideoCapture('../videos/cars.mp4')
#Initializing the yolo model
model=YOLO("../yolo-weights/yolo11n.pt")
#changing the width and height


while True:
    #taking the image from the webcam
    success,image=cap.read()
    defaultMask = cv2.resize(defaultMask, (image.shape[1], image.shape[0]))
    imgRegion=cv2.bitwise_and(image,defaultMask)
    imageGraphics=cv2.imread('graphics.png',cv2.IMREAD_UNCHANGED)
    image=cvzone.overlayPNG(image,imageGraphics,(0,0))
    #taking the result of the image
    results=model(imgRegion,stream=True)
    detections=np.empty((0,5))
    for r in results:
        #looping through all of the boxes
        boxes=r.boxes
        for box in boxes:
            #we can either perform xyxy or xywh depends on our work
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w,h=x2-x1,y2-y1
            #bbox=int(x),int(y),int(w),int(h)
            #print(x1,y1,x2,y2)
            #printing the rectangle across the detected object
            #taking the confidence level and rounding it to 2 decimal places
            conf=math.ceil((box.conf[0]*100))/100
            #print(conf)
            #class name
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=='car' or currentClass=='motorbike' or currentClass=='bus' or currentClass=='truck' and conf>0.3:
                #putting the confidence rectangle of the detected object
                # cvzone.cornerRect(image, (x1, y1, w, h), l=15,rt=5)
                # cvzone.putTextRect(image,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=0.6,
                #                offset=3,thickness=1)
                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)
    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(image, (x1, y1, w, h), l=15, rt=2,colorR=(255,0,0))
        cvzone.putTextRect(image, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2,
                          offset=10, thickness=3)

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx< limits[2] and limits[1]-15<cy<limits[1]+15:
            if totalCount.count(id)==0:
                totalCount.append(id)
                cv2.line(image, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(image, f'Count:{len(totalCount)}', (50,50))
    cv2.putText(image,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("image",image)
    #print("Image shape:",image.shape)
    #print("Mask shape:",defaultMask.shape)
    #cv2.imshow("imageregion",imgRegion)
    cv2.waitKey(1)

