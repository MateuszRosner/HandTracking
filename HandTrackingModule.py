from unittest import result
from xml.etree.ElementInclude import include
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, 
               static_image_mode=False,
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        self.mode = static_image_mode
        self.max_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_conf = min_detection_confidence
        self.min_tracking_conf = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, self.model_complexity,
                     self.min_detection_conf, self.min_tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return results

    def findPosition(self, img, detections, hand, indexes):
        if detections.multi_hand_landmarks:
            for handID, handLms in enumerate(detections.multi_hand_landmarks):
                for id, lm in enumerate(handLms.landmark):
                    if handID in hand and id in indexes:
                        h, w, c = img.shape
                        cx, cy = int(lm.x*w), int(lm.y*h)
                        print(f"ID: {id} x = {cx} y = {cy}")
                        cv2.circle(img, center=(cx, cy), radius=10, 
                                    color=(255,0,0), thickness=cv2.FILLED)

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()    

    while True:
        success, img = cap.read()
        results = detector.findHands(img)
        detector.findPosition(img, results, (0,), (4, 8, 12, 16, 20)) 
        

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), org=(10, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    color=(0, 255, 0), fontScale=1, thickness=1)

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()