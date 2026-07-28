import cv2
import numpy as np
from utils import preprocess_full_res

video_path = "./input/vid1.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame")
        break

    final_img = preprocess_full_res(frame)

    if final_img is not None:
        cv2.imshow('Video Feed', final_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()