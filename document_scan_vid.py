import cv2
import numpy as np
from utils import preprocess_full_res

video_path = "./input/vid1.mp4"
cap = cv2.VideoCapture(video_path)

display_size = (400, 400)
grid_size = (display_size[0] * 2, display_size[1] * 2)  # (width, height) of stacked grid

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps is None:
    fps = 20  # fallback if the source doesn't report fps properly

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./output/output.mp4', fourcc, fps, grid_size)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    img_warped, img_contours, img_canny = preprocess_full_res(frame)

    frame_resized = cv2.resize(frame, display_size)
    contours_resized = cv2.resize(img_contours, display_size)
    canny_bgr = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    canny_resized = cv2.resize(canny_bgr, display_size)

    if img_warped is not None:
        warped_resized = cv2.resize(img_warped, display_size)
    else:
        warped_resized = np.zeros_like(frame_resized)

    top_row = np.hstack((frame_resized, contours_resized))
    bottom_row = np.hstack((canny_resized, warped_resized))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow('Video Feed', grid)
    out.write(grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()