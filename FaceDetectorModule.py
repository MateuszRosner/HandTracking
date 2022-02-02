import cv2
import mediapipe as mp
import time

class faceDetector():
    def __init__(self, model_selection=1, min_detection_confidence=0.8):
        self.model_selection = model_selection
        self.min_det_confidence = min_detection_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.faces = self.mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8)

        self.mp_drawing = mp.solutions.drawing_utils

    def findFaces(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faces.process(imgRGB)

    def drawLandmarks(self, img):
        if self.results.detections:
            for detection in self.results.detections:
                self.mp_drawing.draw_detection(img, detection)


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = faceDetector()    

    while True:
        success, img = cap.read()

        detector.findFaces(img)
        detector.drawLandmarks(img)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), org=(10, 70), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                    color=(0, 255, 0), fontScale=1, thickness=1)

        cv2.imshow("Frame", cv2.flip(img, 1))

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    main()