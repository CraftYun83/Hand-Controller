import cv2
from threading import Thread
import mediapipe as mp
import pydirectinput
import time
class handDetector():
    def __init__(self, mode = False, maxHands = 1, detectionCon = 0.5, trackCon = 0.5, modelComplexity=1):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

def main():

    def left():
        pydirectinput.keyDown('a')
        time.sleep(0.3)
        pydirectinput.keyUp('a')

    def right():
        pydirectinput.keyDown('d')
        time.sleep(0.3)
        pydirectinput.keyUp('d')

    def back():
        pydirectinput.keyDown('s')
        time.sleep(0.3)
        pydirectinput.keyUp('s')

    def forward():
        pydirectinput.keyDown('w')
        time.sleep(0.3)
        pydirectinput.keyUp('w')
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
            if lmlist[4][1] > 480:
                thread = Thread(target = left)
                thread.start()
            elif lmlist[4][1] < 480 and lmlist[4][2] > 460:
                pass
            else:
                thread = Thread(target = right)
                thread.start()
            pass

            if lmlist[4][2] > 260:
                thread = Thread(target = back)
                thread.start()
            elif lmlist[4][2] < 260 and lmlist[4][2] > 240:
                pass
            else:
                thread = Thread(target = forward)
                thread.start()
                

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


main()
