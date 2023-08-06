import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime=0
ctime=0

while True:
    success,img = cap.read()
    #convert to rgb first
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:#for multiple hands
        for handlmk in result.multi_hand_landmarks:
            for id,lm in enumerate(handlmk.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==0:
                    cv.circle(img,(cx,cy),25,(0,255,0),cv.FILLED)

            mpDraw.draw_landmarks(img,handlmk,mpHands.HAND_CONNECTIONS)

# ctime=time.time()
# fps=1/(ctime-ptime)

    cv.imshow('WebCam',img)
    cv.waitKey(1) 
