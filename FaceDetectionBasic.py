import cv2
import mediapipe as mp
import time


# Used a downloaded video from YT
cap = cv2.VideoCapture("Videos/Video1.mp4")
cTime = 0
pTime = 0

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.6)

while True:
    ret_val, image = cap.read()
    # converting to RGB image as face_detection process requires it
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(imageRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # default draw provided by mediapipe
            # mp_drawing.draw_detection(image, detection)
            print(detection.location_data.relative_bounding_box)
            bounding_box = detection.location_data.relative_bounding_box
            # to work with pixel values
            ih, iw, _ = image.shape
            bbox = int(bounding_box.xmin * iw), int(bounding_box.ymin * ih), int(bounding_box.width * iw), int(bounding_box.height * ih)
            cv2.rectangle(image, bbox, (0, 0, 255), 2)
            cv2.putText(image, f"{int(detection.score[0] * 100)}%", (bbox[0] - 5, bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(image, f"FPS {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    cv2.imshow("frame", image)
    cv2.waitKey(1)