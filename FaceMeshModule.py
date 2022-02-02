import cv2
import mediapipe as mp
import time

class faceMesh():
    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.facesMesh = self.mp_face_mesh.FaceMesh(static_image_mode, max_num_faces, refine_landmarks, min_detection_confidence)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def findFacesMesh(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facesMesh.process(imgRGB)

    def drawLandmarks(self, img):
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image=img, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                
                self.mp_drawing.draw_landmarks(image=img, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
                self.mp_drawing.draw_landmarks(image=img, landmark_list=face_landmarks, connections=self.mp_face_mesh.FACEMESH_IRISES,
                                                landmark_drawing_spec=None,
                                                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style())


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = faceMesh()    

    while True:
        success, img = cap.read()

        detector.findFacesMesh(img)
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