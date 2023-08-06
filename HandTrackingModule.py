import cv2 as cv
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands = 2,model_complexity=1,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.model_complexity=model_complexity
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils 
        self.tipIds =[4,8,12,16,20]

    def findHands(self,img,draw=True):
               #convert to rgb first
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)
        # print(result.multi_hand_landmarks)
        if self.result.multi_hand_landmarks:#for multiple hands
            for handlmk in self.result.multi_hand_landmarks:
                 if draw:
                    self.mpDraw.draw_landmarks(img,handlmk,self.mpHands.HAND_CONNECTIONS)
                 
              
        return img
    def findPosition(self,img,handNo=0,draw=True):
        self.lmlist = []
        if self.result.multi_hand_landmarks:
                myHand=self.result.multi_hand_landmarks[handNo]
                for id,lm in enumerate(myHand.landmark):
                    # print(id,lm)
                    h,w,c=img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    # print(id,cx,cy)
                    self.lmlist.append([id,cx,cy])

                    if draw:
                       cv.circle(img,(cx,cy),5,(0,255,0),cv.FILLED)
        return self.lmlist
    
    def fingersUp(self):
        fingers=[]
        #Thumb
        if self.lmlist[self.tipIds[0]][1] < self.lmlist[self.tipIds[0]-1][1]:
              fingers.append(1)
        else:
            fingers.append(0)
        #Other fingers
        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id]-2][2]:
              fingers.append(1)
            else:
                fingers.append(0)
        return fingers
        



        
def main():
    ptime=0
    ctime=0
    cap=cv.VideoCapture(0)
    detector = handDetector()
    while True:
       success,img = cap.read()
       img=detector.findHands(img)
       lmlist=detector.findPosition(img)
       if len(lmlist)!=0:
         print(lmlist[4])
       ctime=time.time()
       fps=1/(ctime-ptime)
       cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

       cv.imshow('WebCam',img)
       cv.waitKey(1) 
if __name__=="__main__":
    main()
