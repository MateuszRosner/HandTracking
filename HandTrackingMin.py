from unittest import result
import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(f"ID: {id} x = {cx} y = {cy}")
                if id % 4 == 0:
                    cv2.circle(img, center=(cx, cy), radius=10, 
                                color=(255,0,0), thickness=cv2.FILLED)
                

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), org=(10, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                color=(0, 255, 0), fontScale=1, thickness=1)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) == ord('q'):
        break