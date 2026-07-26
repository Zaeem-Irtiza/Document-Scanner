import cv2
import numpy as np
from utils import preprocess

video_path = "./input/vid.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    final_img = preprocess(frame)
    cv2.imshow('Webcam Feed', final_img)

    # Wait for 1 millisecond and check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera capture object and close all active OpenCV windows
cap.release()
cv2.destroyAllWindows()