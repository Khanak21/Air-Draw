import cv2 as cv
import numpy as np
import time
import os
import HandTrackingModule as htm

#to import images
folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList=[]
for imPath in myList:
    #read all images
    image=cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header=overlayList[0]
drawColor=(21,21,21)

cap = cv.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = htm.handDetector()

xp,yp=0,0
imgCanvas = np.zeros((720,1280,3),np.uint8)
eraserThickness=100
brushThickness=15
while True:
    #1.import image
    success,img = cap.read()
    img = cv.flip(img,1)
     
    #2. Find hand landmarks
    img = detector.findHands(img,draw=False)
    lmlist = detector.findPosition(img,draw=False)
    if len(lmlist)!=0:
        #tip of index and middle finger
        x1,y1 = lmlist[8][1],lmlist[8][2]
        x2,y2 = lmlist[12][1],lmlist[12][2]

        #3. Check which fingers are up
        fingers=detector.fingersUp()
        # print(fingers)
        #4. If Selection mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp,yp=0,0 #whenever selection,update prev values to zero
            # cv.rectangle(img,(x1,y1-15),(x2,y2+15),drawColor,cv.FILLED)
            if y1 < 125:#if finger inside header
                if 50<x1<150:
                    header=overlayList[0]
                    drawColor=(21,21,21)
                elif 200<x1<300:
                    header=overlayList[1]
                    drawColor=(255,255,255)
                elif 350<x1<400:
                    drawColor=(0,0,255)
                    header=overlayList[2]
                elif 450<x1<550:
                    drawColor=(230,225,92)
                    header=overlayList[3]
                elif 600<x1<700:
                    drawColor=(114,255,193)
                    header=overlayList[4]
                elif 750<x1<850:
                    drawColor=(28,238,250)
                    header=overlayList[5]
                elif 900<x1<1000:
                    drawColor=(196,102,255)
                    header=overlayList[6]
                elif 1050<x1<1100:
                    header=overlayList[7]
                    drawColor=(1,1,1)
        
        # elif fingers[1]==0:
        #     xp,yp=0,0

        #5. If Drawing Mode - index finger is up
        else:
            cv.circle(img,(x1,y1),25,drawColor,cv.FILLED)#dont mention thickness if you wanna fill or t=-1
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(1,1,1):
               cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
               cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
              cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
              cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1

    imgGray=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv = cv.threshold(imgGray,20,255,cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)
    #setting the header image
    img[0:125,0:1280]=header
    cv.imshow("Image",img)
    cv.waitKey(1)



