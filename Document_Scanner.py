import cv2
import numpy as np
from utils import preprocess_full_res

img_orig = cv2.imread("./input/7.jpeg")

img_warped, img_contours, img_canny = preprocess_full_res(img_orig)

display_size = (400, 400)

img_orig_resized = cv2.resize(img_orig, display_size)
img_contours_resized = cv2.resize(img_contours, display_size)
img_canny_bgr = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
img_canny_resized = cv2.resize(img_canny_bgr, display_size)

if img_warped is not None:
    img_warped_resized = cv2.resize(img_warped, display_size)
else:
    img_warped_resized = np.zeros_like(img_orig_resized)

top_row = np.hstack((img_orig_resized, img_contours_resized))
bottom_row = np.hstack((img_canny_resized, img_warped_resized))
grid = np.vstack((top_row, bottom_row))

cv2.imshow("Document Scanner", grid)
cv2.imwrite("./output/output.png", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()